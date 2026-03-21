import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# ── DB connection ─────────────────────────────────────────
DB_USER     = os.getenv("MYSQL_USER", "root")
DB_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
DB_HOST     = os.getenv("MYSQL_HOST", "127.0.0.1")
DB_NAME     = os.getenv("MYSQL_DB", "feedback_intelligence")

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}",
    echo=False
)

# ── Settings ──────────────────────────────────────────────
N_TOPICS    = 10      # number of LDA topics
MAX_FEATURES= 5000    # TF-IDF vocabulary size
MODELS_DIR  = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Load cleaned text from MySQL ──────────────────────────
def load_cleaned_reviews():
    print("Loading cleaned reviews from MySQL ...")
    with engine.connect() as conn:
        df = pd.read_sql("""
            SELECT review_id, cleaned_text
            FROM reviews
            WHERE cleaned_text IS NOT NULL
              AND cleaned_text != ''
        """, conn)
    print(f"  Loaded {len(df):,} cleaned reviews")
    return df

# ── TF-IDF Vectorisation ──────────────────────────────────
def build_tfidf(texts):
    print(f"Building TF-IDF matrix (max_features={MAX_FEATURES}) ...")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),       # unigrams + bigrams
        min_df=5,                 # ignore very rare terms
        max_df=0.90,              # ignore very common terms
        sublinear_tf=True,        # apply log normalization
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Save vectorizer and matrix
    joblib.dump(vectorizer, f"{MODELS_DIR}/tfidf_vectorizer.pkl")
    save_npz(f"{MODELS_DIR}/tfidf_matrix.npz", tfidf_matrix)
    print(f"  Saved to {MODELS_DIR}/tfidf_vectorizer.pkl")
    print(f"  Saved to {MODELS_DIR}/tfidf_matrix.npz")

    return vectorizer, tfidf_matrix

# ── LDA Topic Modelling ───────────────────────────────────
def build_lda(tfidf_matrix, vectorizer):
    print(f"Training LDA model with {N_TOPICS} topics ...")
    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        max_iter=10,
        learning_method="online",   # faster than batch for large data
        random_state=42,
        n_jobs=-1,                  # use all CPU cores
    )
    lda.fit(tfidf_matrix)
    print("  LDA training complete")

    # Save model
    joblib.dump(lda, f"{MODELS_DIR}/lda_model.pkl")
    print(f"  Saved to {MODELS_DIR}/lda_model.pkl")

    return lda

# ── Extract top keywords per topic ───────────────────────
def get_topic_keywords(lda, vectorizer, n_words=10):
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-n_words:][::-1]
        top_words   = [feature_names[i] for i in top_indices]
        topics.append(top_words)
    return topics

def save_topics_to_mysql(topic_keywords):
    print("Saving topics to MySQL ...")
    with engine.begin() as conn:
        # Force clear the table using raw MySQL
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
        conn.execute(text("TRUNCATE TABLE topics"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))

        for topic_id, keywords in enumerate(topic_keywords):
            keywords_str = ", ".join(keywords)
            conn.execute(text("""
                INSERT IGNORE INTO topics (topic_id, top_keywords, topic_label)
                VALUES (:topic_id, :top_keywords, :topic_label)
            """), {
                "topic_id":     topic_id,
                "top_keywords": keywords_str,
                "topic_label":  f"Topic {topic_id}"
            })
    print(f"  {len(topic_keywords)} topics saved to MySQL")

# ── Assign dominant topic to each review ─────────────────
def assign_topics(df, lda, tfidf_matrix, batch_size=5000):
    print("Assigning dominant topic to each review ...")
    n = tfidf_matrix.shape[0]
    topic_ids = []

    for i in range(0, n, batch_size):
        batch = tfidf_matrix[i: i + batch_size]
        doc_topic_dist = lda.transform(batch)
        dominant = np.argmax(doc_topic_dist, axis=1)
        topic_ids.extend(dominant.tolist())
        print(f"  Assigned {min(i+batch_size, n):,} / {n:,} reviews ...")

    # Write back to MySQL
    records = [
        {"review_id": int(rid), "topic_id": int(tid)}
        for rid, tid in zip(df["review_id"].tolist(), topic_ids)
    ]

    with engine.begin() as conn:
        for i in range(0, len(records), batch_size):
            batch = records[i: i + batch_size]
            conn.execute(text("""
                UPDATE reviews
                SET topic_id = :topic_id
                WHERE review_id = :review_id
            """), batch)

    print(f"  topic_id updated for {len(records):,} reviews in MySQL")
    return topic_ids

# ── Print topic summary ───────────────────────────────────
def print_topic_summary(topic_keywords):
    print("\n── LDA Topic Summary ──────────────────────────")
    for i, words in enumerate(topic_keywords):
        print(f"  Topic {i:>2} : {', '.join(words)}")
    print("───────────────────────────────────────────────\n")

# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_cleaned_reviews()

    vectorizer, tfidf_matrix = build_tfidf(df["cleaned_text"].tolist())

    lda = build_lda(tfidf_matrix, vectorizer)

    topic_keywords = get_topic_keywords(lda, vectorizer)
    print_topic_summary(topic_keywords)

    save_topics_to_mysql(topic_keywords)

    assign_topics(df, lda, tfidf_matrix)

    print("topic_model.py complete!")
    print("Your models are saved in the models/ folder.")
    print("MySQL reviews table now has cleaned_text and topic_id filled in.")