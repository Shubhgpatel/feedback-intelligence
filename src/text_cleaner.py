import os
import re
import pandas as pd
import spacy
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

# ── Load spaCy model ──────────────────────────────────────
print("Loading spaCy model ...")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# ── Text cleaning functions ───────────────────────────────
def remove_html(text):
    """Remove HTML tags like <br>, <b>, etc."""
    return re.sub(r"<[^>]+>", " ", str(text))

def remove_urls(text):
    """Remove URLs."""
    return re.sub(r"http\S+|www\S+", " ", text)

def remove_special_chars(text):
    """Keep only letters, numbers and spaces."""
    return re.sub(r"[^a-zA-Z\s]", " ", text)

def normalize_whitespace(text):
    """Collapse multiple spaces into one."""
    return re.sub(r"\s+", " ", text).strip()

def basic_clean(text):
    """Run all basic cleaning steps."""
    text = remove_html(text)
    text = remove_urls(text)
    text = remove_special_chars(text)
    text = normalize_whitespace(text)
    return text.lower()

# ── spaCy lemmatisation pipeline ─────────────────────────
def lemmatize_batch(texts, batch_size=500):
    """
    Takes a list of raw strings.
    Returns a list of cleaned, lemmatised strings.
    """
    cleaned = [basic_clean(t) for t in texts]
    results = []

    print(f"  Running spaCy on {len(cleaned):,} texts ...")
    for i, doc in enumerate(nlp.pipe(cleaned, batch_size=batch_size)):
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop        # remove stopwords
            and not token.is_punct      # remove punctuation
            and not token.is_space      # remove whitespace tokens
            and len(token.lemma_) > 2   # remove very short tokens
        ]
        results.append(" ".join(tokens))

        if (i + 1) % 5000 == 0:
            print(f"    Processed {i+1:,} reviews ...")

    return results

# ── Load reviews from MySQL ───────────────────────────────
def load_reviews():
    print("Loading reviews from MySQL ...")
    with engine.connect() as conn:
        df = pd.read_sql(
            "SELECT review_id, review_text FROM reviews WHERE cleaned_text IS NULL",
            conn
        )
    print(f"  Found {len(df):,} reviews to clean")
    return df

# ── Write cleaned text back to MySQL ─────────────────────
def save_cleaned(review_ids, cleaned_texts, batch_size=1000):
    print("Writing cleaned_text back to MySQL ...")
    records = [
        {"review_id": int(rid), "cleaned_text": ct}
        for rid, ct in zip(review_ids, cleaned_texts)
    ]

    total = len(records)
    saved = 0

    with engine.begin() as conn:
        for i in range(0, total, batch_size):
            batch = records[i: i + batch_size]
            conn.execute(text("""
                UPDATE reviews
                SET cleaned_text = :cleaned_text
                WHERE review_id  = :review_id
            """), batch)
            saved += len(batch)
            print(f"  Updated {saved:,} / {total:,} rows ...")

    print(f"Done — {saved:,} reviews updated in MySQL")

# ── Verify ────────────────────────────────────────────────
def verify():
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM reviews")).scalar()
        cleaned = conn.execute(
            text("SELECT COUNT(*) FROM reviews WHERE cleaned_text IS NOT NULL")
        ).scalar()
        sample = conn.execute(text("""
            SELECT review_id, cleaned_text
            FROM reviews
            WHERE cleaned_text IS NOT NULL
            LIMIT 3
        """)).fetchall()

    print(f"\n── Verification ───────────────────────────────")
    print(f"  Total reviews   : {total:,}")
    print(f"  Cleaned reviews : {cleaned:,}")
    print(f"\n  Sample cleaned texts:")
    for row in sample:
        print(f"    [{row[0]}] {row[1][:120]} ...")
    print(f"───────────────────────────────────────────────\n")

# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_reviews()

    if df.empty:
        print("All reviews already cleaned. Nothing to do.")
    else:
        cleaned_texts = lemmatize_batch(df["review_text"].tolist())
        save_cleaned(df["review_id"].tolist(), cleaned_texts)
        verify()

    print("text_cleaner.py complete!")