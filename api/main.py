from functools import lru_cache
from pathlib import Path
import traceback
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgb_risk_pipeline.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "ml_final_features.csv"

FEATURE_COLS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_rush_hour",
    "length",
    "highway",
    "lanes",
    "maxspeed",
    "oneway",
    "start_node_centrality",
]

app = FastAPI(
    title="Urban Accident Risk Intelligence API",
    description="FastAPI backend for Spatio-Temporal XGBoost risk forecasting, SHAP explainability, and What-If simulation.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None
preprocessor = None
explainer = None
FEATURE_NAMES: list[str] = []
ROADS_DF = pd.DataFrame()
ROADS_LOOKUP: dict[str, dict[str, Any]] = {}


def load_artifacts() -> None:
    global pipeline, preprocessor, explainer, FEATURE_NAMES, ROADS_DF, ROADS_LOOKUP

    pipeline = joblib.load(MODEL_PATH)
    print(f"Pipeline steps found: {list(pipeline.named_steps.keys())}")

    # Dynamically extract preprocessor and final estimator regardless of step names
    preprocessor = pipeline[:-1] if len(pipeline.steps) > 1 else None
    classifier = pipeline.steps[-1][1]

    # Extract feature names dynamically
    FEATURE_NAMES = []
    if preprocessor is not None:
        try:
            FEATURE_NAMES = list(preprocessor.get_feature_names_out())
        except Exception:
            FEATURE_NAMES = []

    # Initialize SHAP TreeExplainer safely
    try:
        explainer = shap.TreeExplainer(classifier)
    except Exception as shap_exc:
        print(f"SHAP TreeExplainer warning: {shap_exc}")
        explainer = None

    raw_df = pd.read_csv(DATA_PATH)
    print(f"CSV columns found: {list(raw_df.columns)}")
    raw_df["road_segment_id"] = raw_df["road_segment_id"].astype(str)

    # Detect target column case-insensitively to compute real historical crash counts
    target_col = next(
        (c for c in raw_df.columns if c.lower() in ("target", "accident", "label", "y")),
        None,
    )
    if target_col:
        hist_counts = (
            raw_df.groupby("road_segment_id")[target_col]
            .sum()
            .rename("historical_accidents")
        )
    else:
        hist_counts = pd.Series(dtype=int, name="historical_accidents")

    static_cols = [
        "road_segment_id",
        "length",
        "highway",
        "lanes",
        "maxspeed",
        "oneway",
        "start_node_centrality",
    ]
    roads = (
        raw_df[static_cols]
        .drop_duplicates(subset=["road_segment_id"])
        .dropna()
        .merge(hist_counts, on="road_segment_id", how="left")
        .reset_index(drop=True)
    )
    roads["historical_accidents"] = roads["historical_accidents"].fillna(0).astype(int)

    ROADS_DF = roads
    ROADS_LOOKUP = roads.set_index("road_segment_id").to_dict(orient="index")
    print(
        f"Loaded model, SHAP explainer ({explainer is not None}), and {len(ROADS_DF)} unique road segments."
    )


try:
    load_artifacts()
except Exception as exc:
    print(f"Startup load failed: {exc}")
    traceback.print_exc()


class RoadPredictionRequest(BaseModel):
    hour: int
    day_of_week: int
    month: int
    is_weekend: int
    is_rush_hour: int
    length: float
    highway: str
    lanes: float
    maxspeed: float
    oneway: int
    start_node_centrality: float


class SimulationRequest(BaseModel):
    baseline: RoadPredictionRequest
    overrides: Dict[str, Any]


def classify_risk(prob: float) -> dict[str, Any]:
    score = round(prob * 100, 1)
    if score >= 80.0:
        tier = "VERY HIGH"
    elif score >= 60.0:
        tier = "HIGH"
    elif score >= 40.0:
        tier = "MEDIUM"
    else:
        tier = "LOW"
    return {
        "risk_probability": round(prob, 4),
        "risk_score": score,
        "risk_tier": tier,
    }


def _is_rush_hour(hour: int, day_of_week: int) -> int:
    return 1 if (day_of_week < 5 and hour in (7, 8, 9, 16, 17, 18)) else 0


@lru_cache(maxsize=256)
def _score_city_cached(hour: int, day_of_week: int, month: int) -> pd.DataFrame:
    batch = ROADS_DF.copy()
    batch["hour"] = hour
    batch["day_of_week"] = day_of_week
    batch["month"] = month
    batch["is_weekend"] = 1 if day_of_week >= 5 else 0
    batch["is_rush_hour"] = _is_rush_hour(hour, day_of_week)

    batch["risk_probability"] = pipeline.predict_proba(batch[FEATURE_COLS])[:, 1]
    return batch.sort_values("risk_probability", ascending=False)


@app.get("/")
def health_check():
    return {
        "status": "online",
        "model_loaded": pipeline is not None,
        "shap_loaded": explainer is not None,
        "total_roads_indexed": len(ROADS_DF),
        "highway_types": sorted(ROADS_DF["highway"].unique().tolist())
        if not ROADS_DF.empty
        else [],
    }


@app.get("/hotspots")
def get_hotspots(
    hour: int = Query(17, ge=0, le=23),
    day_of_week: int = Query(4, ge=0, le=6),
    month: int = Query(10, ge=1, le=12),
    top_n: int = Query(25, ge=5, le=150),
    highway: Optional[str] = Query(None),
):
    if pipeline is None or ROADS_DF.empty:
        raise HTTPException(status_code=500, detail="Model or road catalog not loaded.")

    scored = _score_city_cached(hour, day_of_week, month)
    if highway and highway != "all":
        scored = scored[scored["highway"] == highway]

    probs = scored["risk_probability"].to_numpy()
    summary = {
        "evaluated_roads": int(len(scored)),
        "mean_risk_score": round(float(np.mean(probs) * 100), 1) if len(probs) else 0.0,
        "critical_count": int(np.sum(probs >= 0.80)),
        "high_count": int(np.sum((probs >= 0.60) & (probs < 0.80))),
    }

    top_rows = scored.head(top_n)
    hotspots = []
    for _, row in top_rows.iterrows():
        hotspots.append(
            {
                "road_segment_id": str(row["road_segment_id"]),
                "highway": str(row["highway"]),
                "maxspeed": float(row["maxspeed"]),
                "lanes": float(row["lanes"]),
                "length": round(float(row["length"]), 1),
                "oneway": int(row["oneway"]),
                "start_node_centrality": float(row["start_node_centrality"]),
                "historical_accidents": int(row["historical_accidents"]),
                **classify_risk(float(row["risk_probability"])),
            }
        )

    return {
        "query_time": {"hour": hour, "day_of_week": day_of_week, "month": month},
        "summary": summary,
        "hotspots": hotspots,
    }


@app.get("/timeline/{road_segment_id}")
def get_road_timeline(
    road_segment_id: str,
    day_of_week: int = Query(4, ge=0, le=6),
    month: int = Query(10, ge=1, le=12),
):
    if pipeline is None or road_segment_id not in ROADS_LOOKUP:
        raise HTTPException(status_code=404, detail="Road segment not found.")

    road = ROADS_LOOKUP[road_segment_id]
    rows = [
        {
            "hour": h,
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": 1 if day_of_week >= 5 else 0,
            "is_rush_hour": _is_rush_hour(h, day_of_week),
            "length": road["length"],
            "highway": road["highway"],
            "lanes": road["lanes"],
            "maxspeed": road["maxspeed"],
            "oneway": road["oneway"],
            "start_node_centrality": road["start_node_centrality"],
        }
        for h in range(24)
    ]
    df_24h = pd.DataFrame(rows)
    probs = pipeline.predict_proba(df_24h[FEATURE_COLS])[:, 1]

    return {
        "road_segment_id": road_segment_id,
        "timeline": [
            {"hour": h, "risk_score": round(float(p * 100), 1)}
            for h, p in enumerate(probs)
        ],
    }


@app.post("/predict")
def predict_risk(request: RoadPredictionRequest):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    prob = float(pipeline.predict_proba(pd.DataFrame([request.model_dump()]))[0][1])
    return classify_risk(prob)


@app.post("/explain")
def explain_risk(request: RoadPredictionRequest):
    if pipeline is None or explainer is None:
        raise HTTPException(status_code=500, detail="Model or SHAP explainer not loaded.")

    input_df = pd.DataFrame([request.model_dump()])
    prob = float(pipeline.predict_proba(input_df)[0][1])
    prediction = classify_risk(prob)

    transformed = preprocessor.transform(input_df) if preprocessor is not None else input_df
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    raw_shap = explainer.shap_values(transformed)
    if isinstance(raw_shap, list):
        shap_vals = np.asarray(raw_shap[-1])[0]
    else:
        arr = np.asarray(raw_shap)
        shap_vals = arr[0, :, -1] if arr.ndim == 3 else arr[0]

    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(np.asarray(base_val).ravel()[-1])

    names = FEATURE_NAMES if len(FEATURE_NAMES) == len(shap_vals) else [
        f"Feature {i}" for i in range(len(shap_vals))
    ]

    contributions = []
    for name, val in zip(names, shap_vals):
        clean_name = str(name).replace("num__", "").replace("cat__", "")
        contributions.append(
            {
                "feature": clean_name,
                "shap_value": round(float(val), 4),
                "effect": "increases_risk" if val > 0 else "decreases_risk",
            }
        )

    contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    return {
        **prediction,
        "base_value": round(float(base_val), 4),
        "top_factors": contributions[:7],
    }


@app.post("/simulate")
def simulate_scenario(request: SimulationRequest):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    base_dict = request.baseline.model_dump()
    base_prob = float(pipeline.predict_proba(pd.DataFrame([base_dict]))[0][1])
    base_result = classify_risk(base_prob)

    sim_dict = {**base_dict, **request.overrides}
    if "hour" in request.overrides or "day_of_week" in request.overrides:
        h = int(sim_dict["hour"])
        d = int(sim_dict["day_of_week"])
        sim_dict["is_weekend"] = 1 if d >= 5 else 0
        if "is_rush_hour" not in request.overrides:
            sim_dict["is_rush_hour"] = _is_rush_hour(h, d)

    sim_prob = float(pipeline.predict_proba(pd.DataFrame([sim_dict]))[0][1])
    sim_result = classify_risk(sim_prob)

    return {
        "baseline": base_result,
        "simulated": sim_result,
        "risk_score_change": round(
            sim_result["risk_score"] - base_result["risk_score"], 1
        ),
    }