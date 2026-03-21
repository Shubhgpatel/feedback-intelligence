import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ── DB connection ─────────────────────────────────────────
DB_USER     = os.getenv("MYSQL_USER", "root")
DB_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
DB_HOST     = os.getenv("MYSQL_HOST", "localhost")
DB_NAME     = os.getenv("MYSQL_DB", "feedback_intelligence")

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}",
    echo=False
)

# ── Helpers ───────────────────────────────────────────────
def star_to_sentiment(star):
    if star in [1, 2]:
        return "neg"
    elif star == 3:
        return "neu"
    else:
        return "pos"

def safe_datetime(unix_ts):
    try:
        return datetime.utcfromtimestamp(int(unix_ts))
    except:
        return None

# ── Load CSV ──────────────────────────────────────────────
def load_csv(path="data/raw/Reviews.csv", limit=50000):
    print(f"Reading CSV from {path} ...")
    df = pd.read_csv(path, nrows=limit)
    print(f"Loaded {len(df):,} rows from CSV")
    return df

# ── Clean & transform ─────────────────────────────────────
def transform(df):
    print("Transforming data ...")

    # Rename columns to match our schema
    df = df.rename(columns={
        "Id":             "original_id",
        "ProductId":      "product_id",
        "UserId":         "user_id",
        "ProfileName":    "profile_name",
        "HelpfulnessNumerator":   "helpful_num",
        "HelpfulnessDenominator": "helpful_den",
        "Score":          "star_rating",
        "Time":           "review_time",
        "Summary":        "summary",
        "Text":           "review_text",
    })

    # Drop rows with missing essential fields
    df = df.dropna(subset=["review_text", "star_rating", "product_id"])

    # Map star rating to sentiment label
    df["sentiment_label"] = df["star_rating"].astype(int).apply(star_to_sentiment)

    # Convert Unix timestamp to datetime
    df["review_time"] = df["review_time"].apply(safe_datetime)

    # Helpfulness as string ratio e.g. "4/5"
    df["helpfulness"] = df["helpful_num"].astype(str) + "/" + df["helpful_den"].astype(str)

    # Trim long text fields
    df["summary"]     = df["summary"].astype(str).str[:500]
    df["profile_name"]= df["profile_name"].astype(str).str[:255]

    print(f"After cleaning: {len(df):,} rows")
    return df

# ── Insert products ───────────────────────────────────────
def insert_products(df, conn):
    print("Inserting products ...")
    unique_products = df[["product_id"]].drop_duplicates()

    inserted = 0
    for _, row in unique_products.iterrows():
        try:
            conn.execute(text("""
                INSERT IGNORE INTO products (product_id)
                VALUES (:product_id)
            """), {"product_id": row["product_id"]})
            inserted += 1
        except Exception as e:
            print(f"  Skipped product {row['product_id']}: {e}")

    print(f"  {inserted:,} products inserted")

# ── Insert reviews in batches ─────────────────────────────
def insert_reviews(df, conn, batch_size=1000):
    print("Inserting reviews ...")

    records = df[[
        "product_id", "user_id", "profile_name",
        "helpfulness", "star_rating", "sentiment_label",
        "summary", "review_text", "review_time"
    ]].to_dict(orient="records")

    total = len(records)
    inserted = 0

    for i in range(0, total, batch_size):
        batch = records[i: i + batch_size]
        try:
            conn.execute(text("""
                INSERT INTO reviews
                    (product_id, user_id, profile_name, helpfulness,
                     star_rating, sentiment_label, summary, review_text, review_time)
                VALUES
                    (:product_id, :user_id, :profile_name, :helpfulness,
                     :star_rating, :sentiment_label, :summary, :review_text, :review_time)
            """), batch)
            inserted += len(batch)
            print(f"  Inserted {inserted:,} / {total:,} reviews ...")
        except Exception as e:
            print(f"  Batch error at row {i}: {e}")

    print(f"Done — {inserted:,} reviews inserted into MySQL")

# ── Verify ────────────────────────────────────────────────
def verify(conn):
    print("\n── Verification ──────────────────────────────")
    result = conn.execute(text("SELECT COUNT(*) FROM reviews")).scalar()
    print(f"  reviews table   : {result:,} rows")

    result = conn.execute(text("SELECT COUNT(*) FROM products")).scalar()
    print(f"  products table  : {result:,} rows")

    result = conn.execute(text("""
        SELECT sentiment_label, COUNT(*) as cnt
        FROM reviews
        GROUP BY sentiment_label
    """)).fetchall()
    print("  Sentiment breakdown:")
    for row in result:
        print(f"    {row[0]:>5} → {row[1]:,}")
    print("──────────────────────────────────────────────\n")

# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_csv(limit=50000)   # change limit to load more rows
    df = transform(df)

    with engine.begin() as conn:
        insert_products(df, conn)
        insert_reviews(df, conn)
        verify(conn)

    print("ETL complete! Your MySQL database is ready.")