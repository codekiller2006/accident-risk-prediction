from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "ml_final_features.csv"
MODEL_PATH = BASE_DIR / "models" / "xgb_risk_pipeline.pkl"

NUM_COLS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_rush_hour",
    "length",
    "lanes",
    "maxspeed",
    "oneway",
    "start_node_centrality",
]
CAT_COLS = ["highway"]
FEATURE_COLS = NUM_COLS + CAT_COLS

print("=" * 60)
print("1. DATASET AUDIT (ml_final_features.csv)")
print("=" * 60)
df = pd.read_csv(DATA_PATH)
print(f"Total rows: {len(df):,}")
print(f"Unique road segments: {df['road_segment_id'].nunique():,}")

counts = df["Target"].value_counts().sort_index()
neg, pos = int(counts.get(0, 0)), int(counts.get(1, 0))
ratio = neg / pos if pos else 0
print(f"Class balance -> Target=1 (crashes): {pos:,} | Target=0 (negatives): {neg:,}")
print(f"Exact Pos:Neg ratio -> 1 : {ratio:.2f}")

missing = df.isna().sum()
missing = missing[missing > 0]
print("Missing values in processed CSV:", dict(missing) if len(missing) else "None (0 missing)")

df["parsed_time"] = pd.to_datetime(df["Start_Time"], format="mixed", errors="coerce")
print(f"Date range: {df['parsed_time'].min()}  -->  {df['parsed_time'].max()}")
print("\nNumeric summary (min / median / max):")
for col in ["length", "lanes", "maxspeed", "start_node_centrality"]:
    print(f"  {col:22s}: {df[col].min():.4g} / {df[col].median():.4g} / {df[col].max():.4g}")

print("\n" + "=" * 60)
print("2. SAVED PIPELINE AUDIT (xgb_risk_pipeline.pkl)")
print("=" * 60)
saved_pipe = joblib.load(MODEL_PATH)
print("Pipeline steps:", saved_pipe.named_steps)
xgb_model = saved_pipe.steps[-1][1]
params = xgb_model.get_params()
for k in ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "scale_pos_weight", "reg_alpha", "reg_lambda"]:
    print(f"  {k:18s}: {params.get(k)}")

print("\n" + "=" * 60)
print("3. LEAKAGE-FREE 3-MODEL BENCHMARK (Chronological 80/20 Split)")
print("=" * 60)
df_clean = df.dropna(subset=FEATURE_COLS + ["Target", "parsed_time"]).sort_values("parsed_time")
split_idx = int(len(df_clean) * 0.8)
train_df = df_clean.iloc[:split_idx]
test_df = df_clean.iloc[split_idx:]

print(f"Train rows: {len(train_df):,} (up to {train_df['parsed_time'].max().date()})")
print(f"Test rows:  {len(test_df):,} (from {test_df['parsed_time'].min().date()})")

X_train, y_train = train_df[FEATURE_COLS], train_df["Target"].astype(int)
X_test, y_test = test_df[FEATURE_COLS], test_df["Target"].astype(int)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ]
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=14, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    ),
}

print(f"\n{'Model':22s} | {'ROC-AUC':8s} | {'PR-AUC':8s} | {'F1':8s} | {'Prec':8s} | {'Recall':8s}")
print("-" * 74)
for name, clf in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", clf)])
    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    print(
        f"{name:22s} | "
        f"{roc_auc_score(y_test, probs):8.4f} | "
        f"{average_precision_score(y_test, probs):8.4f} | "
        f"{f1_score(y_test, preds):8.4f} | "
        f"{precision_score(y_test, preds):8.4f} | "
        f"{recall_score(y_test, preds):8.4f}"
    )