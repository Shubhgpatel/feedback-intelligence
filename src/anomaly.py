import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
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

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Load daily sentiment aggregates from MySQL ────────────
def load_daily_aggregates():
    print("Loading daily sentiment aggregates from MySQL ...")
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT
                DATE(review_time)                          AS review_date,
                COUNT(*)                                   AS total_reviews,
                SUM(CASE WHEN sentiment_label = 'pos' THEN 1 ELSE 0 END) AS pos_count,
                SUM(CASE WHEN sentiment_label = 'neg' THEN 1 ELSE 0 END) AS neg_count,
                SUM(CASE WHEN sentiment_label = 'neu' THEN 1 ELSE 0 END) AS neu_count,
                ROUND(AVG(star_rating), 4)                 AS avg_star_rating,
                ROUND(
                    SUM(CASE WHEN sentiment_label = 'neg' THEN 1 ELSE 0 END)
                    * 100.0 / COUNT(*), 4
                )                                          AS neg_ratio
            FROM reviews
            WHERE review_time IS NOT NULL
            GROUP BY DATE(review_time)
            ORDER BY review_date ASC
        """), conn)

    print(f"  Loaded {len(df):,} daily data points")
    return df

# ── Build features for IsolationForest ───────────────────
def build_features(df):
    """
    Features used for anomaly detection:
    - total_reviews   : spike in volume could signal a viral event
    - neg_ratio       : sudden spike in negative reviews
    - avg_star_rating : sudden drop in average rating
    - neg_count       : raw count of negative reviews
    """
    features = df[[
        "total_reviews",
        "neg_ratio",
        "avg_star_rating",
        "neg_count",
    ]].copy()

    # Fill any NaN values
    features = features.fillna(0)
    return features.values

# ── Train IsolationForest ─────────────────────────────────
def train_isolation_forest(X):
    print("Training IsolationForest anomaly detector ...")
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,   # expect ~5% of days to be anomalous
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)
    print("  IsolationForest training complete")

    # Save model
    model_path = f"{MODELS_DIR}/anomaly_detector.pkl"
    joblib.dump(model, model_path)
    print(f"  Saved to {model_path}")
    return model

# ── Detect anomalies ──────────────────────────────────────
def detect_anomalies(model, X, df):
    """
    IsolationForest returns:
      -1 = anomaly
       1 = normal
    We convert to anomaly_flag: 1 = anomaly, 0 = normal
    """
    predictions  = model.predict(X)
    anomaly_flags = (predictions == -1).astype(int)
    scores        = model.score_samples(X)   # lower = more anomalous

    df = df.copy()
    df["anomaly_flag"]  = anomaly_flags
    df["anomaly_score"] = scores

    total_days     = len(df)
    anomaly_days   = anomaly_flags.sum()

    print(f"\n── Anomaly Detection Results ──────────────────")
    print(f"  Total days analysed : {total_days}")
    print(f"  Anomalous days      : {anomaly_days}")
    print(f"  Normal days         : {total_days - anomaly_days}")
    print(f"\n  Anomalous dates:")

    anomalies = df[df["anomaly_flag"] == 1].sort_values("anomaly_score")
    for _, row in anomalies.head(10).iterrows():
        print(f"    {row['review_date']}  "
              f"total={int(row['total_reviews']):>5}  "
              f"neg_ratio={row['neg_ratio']:>6.2f}%  "
              f"avg_stars={row['avg_star_rating']:.2f}  "
              f"score={row['anomaly_score']:.4f}")
    print(f"───────────────────────────────────────────────\n")

    return df

# ── Save anomaly flags to predictions table ───────────────
def save_anomaly_flags(df_daily):
    """
    For each anomalous date, update all predictions for reviews
    on that date to have anomaly_flag = 1
    """
    print("Saving anomaly flags to predictions table ...")

    anomalous_dates = df_daily[df_daily["anomaly_flag"] == 1]["review_date"].tolist()
    anomalous_dates = [str(d) for d in anomalous_dates]

    if not anomalous_dates:
        print("  No anomalies found — nothing to update")
        return

    with engine.begin() as conn:
        # Reset all flags first
        conn.execute(text("UPDATE predictions SET anomaly_flag = 0"))

        # Set flag for anomalous dates
        for date_str in anomalous_dates:
            conn.execute(text("""
                UPDATE predictions p
                JOIN reviews r ON p.review_id = r.review_id
                SET p.anomaly_flag = 1
                WHERE DATE(r.review_time) = :review_date
            """), {"review_date": date_str})

    # Count total flagged predictions
    with engine.connect() as conn:
        flagged = conn.execute(
            text("SELECT COUNT(*) FROM predictions WHERE anomaly_flag = 1")
        ).scalar()

    print(f"  Updated {flagged:,} predictions with anomaly_flag = 1")
    print(f"  Covering {len(anomalous_dates)} anomalous dates")

# ── Save daily aggregates to MySQL ────────────────────────
def save_daily_aggregates(df_daily):
    """
    Save the daily aggregate + anomaly flag to a summary table
    for easy dashboard querying.
    """
    print("Saving daily aggregates to MySQL ...")

    with engine.begin() as conn:
        # Create table if not exists
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS daily_sentiment (
                id             INT AUTO_INCREMENT PRIMARY KEY,
                review_date    DATE NOT NULL UNIQUE,
                total_reviews  INT,
                pos_count      INT,
                neg_count      INT,
                neu_count      INT,
                avg_star_rating FLOAT,
                neg_ratio      FLOAT,
                anomaly_flag   TINYINT DEFAULT 0,
                anomaly_score  FLOAT,
                created_at     DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))

        # Clear and reload
        conn.execute(text("DELETE FROM daily_sentiment"))

        records = df_daily[[
            "review_date", "total_reviews", "pos_count",
            "neg_count", "neu_count", "avg_star_rating",
            "neg_ratio", "anomaly_flag", "anomaly_score"
        ]].to_dict(orient="records")

        # Convert date objects to strings
        for r in records:
            r["review_date"]    = str(r["review_date"])
            r["anomaly_score"]  = float(r["anomaly_score"])
            r["anomaly_flag"]   = int(r["anomaly_flag"])

        conn.execute(text("""
            INSERT INTO daily_sentiment
                (review_date, total_reviews, pos_count, neg_count,
                 neu_count, avg_star_rating, neg_ratio,
                 anomaly_flag, anomaly_score)
            VALUES
                (:review_date, :total_reviews, :pos_count, :neg_count,
                 :neu_count, :avg_star_rating, :neg_ratio,
                 :anomaly_flag, :anomaly_score)
        """), records)

    print(f"  Saved {len(records):,} daily records to daily_sentiment table")

# ── Print summary ─────────────────────────────────────────
def print_summary():
    with engine.connect() as conn:
        total_anomalies = conn.execute(
            text("SELECT COUNT(*) FROM daily_sentiment WHERE anomaly_flag = 1")
        ).scalar()
        worst_day = conn.execute(text("""
            SELECT review_date, neg_ratio, total_reviews, avg_star_rating
            FROM daily_sentiment
            WHERE anomaly_flag = 1
            ORDER BY neg_ratio DESC
            LIMIT 1
        """)).fetchone()

    print(f"\n── Final Summary ──────────────────────────────")
    print(f"  Anomalous days detected : {total_anomalies}")
    if worst_day:
        print(f"  Worst anomaly day       : {worst_day[0]}")
        print(f"    neg_ratio             : {worst_day[1]:.2f}%")
        print(f"    total_reviews         : {worst_day[2]}")
        print(f"    avg_star_rating       : {worst_day[3]:.2f}")
    print(f"───────────────────────────────────────────────\n")

# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Load daily aggregates
    df_daily = load_daily_aggregates()

    if len(df_daily) < 10:
        print("Not enough daily data points to run anomaly detection.")
        print("Need at least 10 days of data.")
        exit(1)

    # 2. Build features
    X = build_features(df_daily)

    # 3. Train IsolationForest
    model = train_isolation_forest(X)

    # 4. Detect anomalies
    df_daily = detect_anomalies(model, X, df_daily)

    # 5. Save daily aggregates with anomaly flags
    save_daily_aggregates(df_daily)

    # 6. Update predictions table with anomaly flags
    save_anomaly_flags(df_daily)

    # 7. Print final summary
    print_summary()

    print("anomaly.py complete!")
    print("Anomaly flags saved to both daily_sentiment and predictions tables.")
    