import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
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

# ── Label encoding ────────────────────────────────────────
# pos=2, neu=1, neg=0
LABEL_MAP     = {"neg": 0, "neu": 1, "pos": 2}
LABEL_INVERSE = {0: "neg", 1: "neu", 2: "pos"}

# ── Load data from MySQL ──────────────────────────────────
def load_data():
    print("Loading reviews from MySQL ...")
    with engine.connect() as conn:
        df = pd.read_sql("""
            SELECT review_id, cleaned_text, sentiment_label
            FROM reviews
            WHERE cleaned_text IS NOT NULL
              AND cleaned_text != ''
              AND sentiment_label IS NOT NULL
        """, conn)
    print(f"  Loaded {len(df):,} reviews")
    return df

# ── Load TF-IDF matrix ────────────────────────────────────
def load_tfidf():
    path = f"{MODELS_DIR}/tfidf_matrix.npz"
    print(f"Loading TF-IDF matrix from {path} ...")
    matrix = load_npz(path)
    print(f"  Matrix shape: {matrix.shape}")
    return matrix

# ── Prepare labels ────────────────────────────────────────
def prepare_labels(df):
    df["label"] = df["sentiment_label"].map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    print(f"  Label distribution:")
    print(f"    neg : {(df['label']==0).sum():,}")
    print(f"    neu : {(df['label']==1).sum():,}")
    print(f"    pos : {(df['label']==2).sum():,}")
    return df

# ── Train / evaluate helper ───────────────────────────────
def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    f1    = f1_score(y_test, preds, average="macro")
    print(f"\n── {name} Results ──────────────────────────────")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, preds, target_names=["neg","neu","pos"]))
    print(f"  Confusion Matrix:")
    cm = confusion_matrix(y_test, preds)
    print(f"    {'':>6} neg   neu   pos")
    for i, row in enumerate(cm):
        label = ["neg","neu","pos"][i]
        print(f"    {label:>6} {row[0]:>5} {row[1]:>5} {row[2]:>5}")
    print(f"───────────────────────────────────────────────")
    return acc, f1, preds

# ── Train Logistic Regression ─────────────────────────────
def train_logistic(X_train, y_train):
    print("\nTraining Logistic Regression ...")
    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",  # handles class imbalance
        solver="saga",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    print("  Logistic Regression training complete")
    return model

# ── Train XGBoost ─────────────────────────────────────────
def train_xgboost(X_train, y_train):
    print("\nTraining XGBoost ...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
        tree_method="hist",       # fast histogram method
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    print("  XGBoost training complete")
    return model

# ── Save predictions to MySQL ─────────────────────────────
def save_predictions(df, best_model, X, batch_size=2000):
    print("\nSaving predictions to MySQL ...")

    # Get predicted labels and confidence scores
    preds       = best_model.predict(X)
    proba       = best_model.predict_proba(X)
    confidences = np.max(proba, axis=1)

    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        records.append({
            "review_id":      int(row["review_id"]),
            "sentiment_label": LABEL_INVERSE[int(preds[i])],
            "confidence":      float(round(confidences[i], 4)),
            "anomaly_flag":    0,
        })

    total   = len(records)
    saved   = 0

    with engine.begin() as conn:
        # Clear old predictions first
        conn.execute(text("DELETE FROM predictions"))

        for i in range(0, total, batch_size):
            batch = records[i: i + batch_size]
            conn.execute(text("""
                INSERT INTO predictions
                    (review_id, sentiment_label, confidence, anomaly_flag)
                VALUES
                    (:review_id, :sentiment_label, :confidence, :anomaly_flag)
            """), batch)
            saved += len(batch)
            print(f"  Saved {saved:,} / {total:,} predictions ...")

    print(f"  Done — {saved:,} predictions written to MySQL")

# ── Verify predictions ────────────────────────────────────
def verify_predictions():
    with engine.connect() as conn:
        total = conn.execute(
            text("SELECT COUNT(*) FROM predictions")
        ).scalar()
        avg_conf = conn.execute(
            text("SELECT AVG(confidence) FROM predictions")
        ).scalar()
        breakdown = conn.execute(text("""
            SELECT sentiment_label, COUNT(*) as cnt
            FROM predictions
            GROUP BY sentiment_label
        """)).fetchall()

    print(f"\n── Predictions Verification ───────────────────")
    print(f"  Total predictions : {total:,}")
    print(f"  Avg confidence    : {avg_conf:.4f}")
    print(f"  Breakdown:")
    for row in breakdown:
        print(f"    {row[0]:>5} → {row[1]:,}")
    print(f"───────────────────────────────────────────────\n")

# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load data
    df     = load_data()
    df     = prepare_labels(df)
    X_full = load_tfidf()

    # Align rows — tfidf matrix rows must match df rows
    X = X_full[:len(df)]
    y = df["label"].values

    # 2. Train / test split (80/20 stratified)
    print("\nSplitting data 80/20 ...")
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)),
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    print(f"  Train : {X_train.shape[0]:,} samples")
    print(f"  Test  : {X_test.shape[0]:,} samples")

    # Balance the training data
    print("\nBalancing classes with SMOTE ...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"  Resampled train size: {X_train.shape[0]:,}")

    # 3. Train both models
    lr_model  = train_logistic(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    # 4. Evaluate both
    lr_acc,  lr_f1,  _ = evaluate("Logistic Regression", lr_model,  X_test, y_test)
    xgb_acc, xgb_f1, _ = evaluate("XGBoost",             xgb_model, X_test, y_test)

    # 5. Pick best model
    if xgb_f1 >= lr_f1:
        best_model      = xgb_model
        best_model_name = "XGBoost"
    else:
        best_model      = lr_model
        best_model_name = "Logistic Regression"

    print(f"\n🏆 Best model: {best_model_name} (Macro F1 = {max(xgb_f1, lr_f1):.4f})")

    # 6. Save best model
    model_path = f"{MODELS_DIR}/best_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"   Saved to {model_path}")

    # Also save both individually
    joblib.dump(lr_model,  f"{MODELS_DIR}/logistic_regression.pkl")
    joblib.dump(xgb_model, f"{MODELS_DIR}/xgboost_model.pkl")
    print(f"   All models saved to {MODELS_DIR}/")

    # 7. Save predictions to MySQL
    save_predictions(df, best_model, X)
    verify_predictions()

    print("train_model.py complete!")
    print("Your predictions table in MySQL is now fully populated.")