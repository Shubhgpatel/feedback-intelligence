# Customer Feedback Intelligence System

An end-to-end machine learning pipeline that ingests customer reviews, stores them in a relational database, applies NLP and sentiment analysis, and surfaces insights through an interactive web dashboard with a conversational AI interface.

---

## What it does

The system takes raw customer reviews from the Amazon Fine Food Reviews dataset and runs them through a full data science pipeline. Reviews are cleaned and stored in MySQL, processed with NLP techniques to extract topics and sentiment, scored by a trained machine learning classifier, and made queryable through a LangChain agent that accepts plain English questions. The final output is a Streamlit dashboard where anyone can explore the data, ask questions, and view anomaly alerts without writing a single line of SQL.

---

## Tech stack

- **Python 3.11** — core language for all pipeline scripts
- **MySQL** — relational data warehouse storing reviews, predictions, topics, and daily aggregates
- **spaCy** — text preprocessing, tokenisation, lemmatisation, stopword removal
- **scikit-learn** — TF-IDF vectorisation, LDA topic modelling, Logistic Regression classifier, IsolationForest anomaly detection
- **XGBoost** — secondary sentiment classifier
- **LangChain** — SQL agent that converts natural language questions into MySQL queries
- **Groq API** — runs llama-3.1-8b-instant for fast, free LLM inference
- **Streamlit** — interactive web dashboard
- **Plotly** — charts and visualisations
- **Railway** — cloud MySQL hosting
- **Streamlit Cloud** — app deployment

---

## Project structure

```
feedback-intelligence/
├── app.py                    # Streamlit dashboard (entry point)
├── src/
│   ├── etl.py                # Loads Reviews.csv into MySQL
│   ├── text_cleaner.py       # NLP preprocessing with spaCy
│   ├── topic_model.py        # TF-IDF + LDA topic modelling
│   ├── train_model.py        # Sentiment classifier training
│   ├── langchain_agent.py    # LangChain SQL agent
│   └── anomaly.py            # IsolationForest anomaly detection
├── models/                   # Saved .pkl model files
├── data/
│   └── raw/                  # Raw CSV files (not committed)
├── schema.sql                # MySQL table definitions
├── requirements.txt
└── .env                      # Local environment variables (not committed)
```

---

## Database schema

The warehouse uses four tables in third normal form.

**reviews** — core fact table. Stores raw and cleaned review text, star rating, sentiment label, topic assignment, and review timestamp.

**predictions** — ML output per review. Stores predicted sentiment label, confidence score, and anomaly flag.

**topics** — LDA topic registry. Stores top keywords and a human-readable label per topic cluster.

**products** — master list of unique products referenced by reviews.

**daily_sentiment** — aggregated table created by the anomaly detection script. Stores daily review counts, sentiment breakdowns, average star rating, and anomaly scores.

---

## Pipeline overview

**Phase 1 — Data ingestion**
Reads the Amazon Fine Food Reviews CSV and normalises it into JSON files. Supports optional Reddit ingestion via PRAW.

**Phase 2 — ETL and data warehouse**
Transforms raw records and bulk-inserts them into MySQL using SQLAlchemy. Maps star ratings to sentiment labels (1-2 = neg, 3 = neu, 4-5 = pos).

**Phase 3 — NLP preprocessing**
Runs each review through a spaCy pipeline to remove HTML, URLs, punctuation, and stopwords, then lemmatises the remaining tokens. Writes cleaned text back to MySQL. Builds a TF-IDF matrix and trains an LDA model to assign each review a dominant topic cluster.

**Phase 4A — Sentiment classification**
Trains Logistic Regression and XGBoost classifiers on the TF-IDF features. Uses SMOTE to balance the heavily skewed class distribution. Evaluates on a stratified 80/20 split and saves the best model by macro F1 score. Writes prediction scores and confidence values to the predictions table.

**Phase 4B — LangChain SQL agent**
Builds a two-step LangChain chain. The first step uses the LLM to generate a valid MySQL query from a plain English question. The second step executes that query against the real database and passes the results back to the LLM to generate a business-readable answer. This avoids hallucination by grounding every response in real query output.

**Phase 4C — Anomaly detection**
Aggregates reviews by day and builds a feature matrix of total volume, negative ratio, average star rating, and raw negative count. Trains an IsolationForest with 5% contamination to flag statistically unusual days. Writes anomaly flags back to both the predictions and daily_sentiment tables.

**Phase 5 — Dashboard**
Four-page Streamlit app: an overview dashboard with KPI cards and trend charts, an AI analyst chat interface backed by the LangChain agent, a filterable review explorer, and an anomaly timeline showing flagged dates.

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/Shubhgpatel/feedback-intelligence.git
cd feedback-intelligence
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**4. Set up environment variables**

Create a `.env` file in the project root:
```
MYSQL_HOST=your-mysql-host
MYSQL_USER=your-mysql-user
MYSQL_PASSWORD=your-mysql-password
MYSQL_DB=your-database-name
MYSQL_PORT=3306
GROQ_API_KEY=your-groq-api-key
```

**5. Create the database schema**
```bash
mysql -u root -p < schema.sql
```

**6. Download the dataset**

Download the Amazon Fine Food Reviews dataset from Kaggle and place `Reviews.csv` in `data/raw/`.

**7. Run the pipeline in order**
```bash
python src/etl.py
python src/text_cleaner.py
python src/topic_model.py
python src/train_model.py
python src/anomaly.py
```

**8. Launch the dashboard**
```bash
streamlit run app.py
```

---

## Model performance

Trained on 50,000 reviews with SMOTE oversampling to correct class imbalance (77% positive, 15% negative, 8% neutral).

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 78.4% | 0.62 |
| XGBoost | 80.9% | 0.57 |

Logistic Regression was selected as the production model based on higher macro F1, which better reflects performance across all three classes including the minority neutral class.

---

## Deployment

The app is deployed on Streamlit Cloud with the MySQL database hosted on Railway. The LangChain agent uses the Groq API for LLM inference, which runs llama-3.1-8b-instant in the cloud and requires no local model setup.

Live demo: https://feedback-intelligence-kkwqedustr3smaolx9wwkc.streamlit.app

---

## Notes

The `.env` file, `data/raw/` folder, and all `.pkl` and `.npz` model files are excluded from version control. If you clone this repo you will need to re-run the full pipeline to regenerate the models, or download pre-trained models separately.

The LangChain agent occasionally generates imperfect SQL for complex aggregation queries when using smaller LLMs. Adding explicit SQL examples to the schema prompt in `langchain_agent.py` improves reliability for specific query patterns.
