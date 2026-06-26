import os
import sys
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

DB_USER      = os.getenv("MYSQL_USER", "root")
DB_PASSWORD  = os.getenv("MYSQL_PASSWORD", "")
DB_HOST      = os.getenv("MYSQL_HOST", "127.0.0.1")
DB_NAME      = os.getenv("MYSQL_DB", "feedback_intelligence")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{MYSQL_PORT}/{DB_NAME}"
engine = create_engine(DB_URL, echo=False)

SCHEMA_CONTEXT = """
You are a MySQL expert. Given a question, write a single valid MySQL query.

Database: feedback_intelligence

Tables:

reviews (review_id, product_id, user_id, profile_name, helpfulness,
         star_rating, sentiment_label, summary, review_text, cleaned_text,
         topic_id, review_time, created_at)
  - sentiment_label values: 'pos', 'neu', 'neg'
  - star_rating: 1 to 5

predictions (prediction_id, review_id, sentiment_label, confidence, anomaly_flag)
  - confidence: float 0.0 to 1.0
  - anomaly_flag: 0 or 1

products (product_id, created_at)

topics (topic_id, top_keywords, topic_label, created_at)
  - top_keywords: comma separated words e.g. 'tea, coffee, cup, drink'

Rules:
- Write ONLY the SQL query, nothing else
- No markdown, no backticks, no explanation
- Always end with a semicolon
- Use LIMIT 20 unless the question asks for all
- ALWAYS include a FROM clause in every query
- For sentiment percentages use this exact pattern:
  SELECT ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM reviews), 2) AS percentage
  FROM reviews WHERE sentiment_label = 'pos';
- sentiment_label values are exactly: 'pos', 'neu', 'neg'
- Never use CASE WHEN for simple WHERE filters
- For topic counts ALWAYS join reviews with topics like this:
  SELECT t.topic_label, t.top_keywords, COUNT(r.review_id) AS review_count
  FROM reviews r JOIN topics t ON r.topic_id = t.topic_id
  GROUP BY t.topic_id, t.topic_label
  ORDER BY review_count DESC LIMIT 3;
"""

SQL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SCHEMA_CONTEXT),
    ("human", "Question: {question}\n\nWrite only the MySQL query:"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful data analyst.
Given a question and the real SQL query results,
give a clear and concise business insight answer.
Be direct. Use the actual numbers from the results.
Do not make up numbers."""),
    ("human", """Question: {question}

SQL Query used: {sql}

Query Results:
{results}

Give a clear answer based on these real results:"""),
])

def build_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

def execute_sql(sql):
    sql = sql.strip().replace("```sql", "").replace("```", "").strip()
    if not sql.endswith(";"):
        sql += ";"
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
        if df.empty:
            return "No results found.", df
        return df.to_string(index=False), df
    except Exception as e:
        return f"SQL Error: {str(e)}", None

def ask(question, llm, verbose=True):
    if verbose:
        print(f"\n[1] Generating SQL for: {question}")
    sql_chain = SQL_PROMPT | llm | StrOutputParser()
    sql = sql_chain.invoke({"question": question})
    sql = sql.strip().replace("```sql","").replace("```","").strip()
    if verbose:
        print(f"[2] Generated SQL:\n    {sql}")
    if verbose:
        print(f"[3] Executing against MySQL ...")
    results_str, df = execute_sql(sql)
    if verbose:
        print(f"[4] Raw Results:\n{results_str}\n")
    if verbose:
        print(f"[5] Generating insight ...")
    answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
    answer = answer_chain.invoke({
        "question": question,
        "sql":      sql,
        "results":  results_str,
    })
    return sql, results_str, answer

def chat_loop():
    print("=" * 60)
    print("  Customer Feedback Intelligence — AI Analyst")
    print("  Powered by LangChain + Ollama + MySQL")
    print("=" * 60)
    print("\nType your question in plain English.")
    print("Type 'quit' to exit.\n")
    llm = build_llm()
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        try:
            sql, results, answer = ask(user_input, llm, verbose=False)
            print(f"\nSQL    : {sql}")
            print(f"Result : {results[:300]}")
            print(f"\nAgent  : {answer}\n")
            print("-" * 60)
        except Exception as e:
            print(f"Error: {e}\n")

def run_quick_test():
    print("\nRunning quick test queries ...\n")
    llm = build_llm()
    test_questions = [
        "How many total reviews are in the database?",
        "What percentage of reviews are positive?",
        "What are the top 3 topics by number of reviews?",
    ]
    for q in test_questions:
        print(f"Q: {q}")
        sql, results, answer = ask(q, llm, verbose=True)
        print(f"\nFinal Answer: {answer}")
        print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_quick_test()
    else:
        chat_loop()