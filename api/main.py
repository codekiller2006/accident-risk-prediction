from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "ml_final_features.csv"
COORDS_PATH = BASE_DIR / "data" / "processed" / "road_coordinates.csv"
MODEL_PATH = BASE_DIR / "models" / "xgb_risk_pipeline.pkl"

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

# Global in-memory artifacts loaded once at startup
pipeline = None
preprocessor = None
xgb_model = None
shap_explainer = None
roads_df: Optional[pd.DataFrame] = None
roads_lookup: Dict[str, Dict[str, Any]] = {}
highway_types: List[str] = []
coordinates_loaded: bool = False
mapped_roads_count: int = 0


def _is_rush_hour(hour: int, day_of_week: int) -> int:
    return 1 if (day_of_week < 5 and hour in (7, 8, 9, 16, 17, 18)) else 0


def classify_risk(prob: float) -> Dict[str, Any]:
    score = round(float(prob) * 100.0, 1)
    if score >= 80.0:
        tier = "VERY HIGH"
    elif score >= 60.0:
        tier = "HIGH"
    elif score >= 40.0:
        tier = "MEDIUM"
    else:
        tier = "LOW"
    return {
        "probability": round(float(prob), 4),
        "risk_score": score,
        "risk_tier": tier,
    }


def _clean_feature_name(raw_name: str) -> str:
    if "__" in raw_name:
        return raw_name.split("__", 1)[1]
    return raw_name


def load_artifacts() -> None:
    global pipeline, preprocessor, xgb_model, shap_explainer
    global roads_df, roads_lookup, highway_types
    global coordinates_loaded, mapped_roads_count

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model pipeline not found at {MODEL_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Feature dataset not found at {DATA_PATH}")

    # 1. Load trained sklearn + XGBoost pipeline
    pipeline = joblib.load(MODEL_PATH)
    step_names = list(pipeline.named_steps.keys())
    print(f"Pipeline steps found: {step_names}")

    preprocessor = pipeline.named_steps.get("preprocessor")
    xgb_model = pipeline.named_steps.get("model", pipeline.steps[-1][1])

    # 2. Initialize SHAP TreeExplainer once
    try:
        shap_explainer = shap.TreeExplainer(xgb_model)
    except Exception as exc:
        print(f"Warning: Could not initialize SHAP TreeExplainer: {exc}")
        shap_explainer = None

    # 3. Load and deduplicate Austin road segments + historical crash totals
    raw_df = pd.read_csv(DATA_PATH)
    print(f"CSV columns found: {list(raw_df.columns)}")
    raw_df["road_segment_id"] = raw_df["road_segment_id"].astype(str)

    crash_counts = (
        raw_df.groupby("road_segment_id")["Target"]
        .sum()
        .astype(int)
        .rename("historical_accidents")
        .reset_index()
    )

    static_cols = [
        "road_segment_id",
        "length",
        "highway",
        "lanes",
        "maxspeed",
        "oneway",
        "start_node_centrality",
    ]
    unique_roads = (
        raw_df[static_cols]
        .dropna()
        .drop_duplicates(subset=["road_segment_id"])
        .merge(crash_counts, on="road_segment_id", how="left")
    )
    unique_roads["historical_accidents"] = (
        unique_roads["historical_accidents"].fillna(0).astype(int)
    )

    # 4. Merge real OpenStreetMap coordinates & street names if generated
    if COORDS_PATH.exists():
        coords_df = pd.read_csv(COORDS_PATH)
        coords_df["road_segment_id"] = coords_df["road_segment_id"].astype(str)
        coords_df = coords_df.drop_duplicates(subset=["road_segment_id"])
        for col in ("lat", "lon"):
            if col in coords_df.columns:
                coords_df[col] = pd.to_numeric(coords_df[col], errors="coerce")
        if "road_name" not in coords_df.columns:
            coords_df["road_name"] = ""
        coords_df["road_name"] = coords_df["road_name"].fillna("").astype(str).str.strip()

        unique_roads = unique_roads.merge(
            coords_df[["road_segment_id", "lat", "lon", "road_name"]],
            on="road_segment_id",
            how="left",
        )
        unique_roads["road_name"] = unique_roads["road_name"].fillna("")
        unique_roads["has_exact_coords"] = (
            unique_roads["lat"].notna() & unique_roads["lon"].notna()
        )
        mapped_roads_count = int(unique_roads["has_exact_coords"].sum())
        coordinates_loaded = mapped_roads_count > 0
    else:
        unique_roads["lat"] = np.nan
        unique_roads["lon"] = np.nan
        unique_roads["road_name"] = ""
        unique_roads["has_exact_coords"] = False
        mapped_roads_count = 0
        coordinates_loaded = False

    roads_df = unique_roads.reset_index(drop=True)
    highway_types = sorted(roads_df["highway"].astype(str).unique().tolist())

    roads_lookup = {}
    for row in roads_df.to_dict(orient="records"):
        lat_val = float(row["lat"]) if pd.notna(row["lat"]) else None
        lon_val = float(row["lon"]) if pd.notna(row["lon"]) else None
        roads_lookup[row["road_segment_id"]] = {
            "road_segment_id": row["road_segment_id"],
            "road_name": str(row["road_name"]),
            "lat": lat_val,
            "lon": lon_val,
            "has_exact_coords": bool(row["has_exact_coords"]),
            "length": round(float(row["length"]), 1),
            "highway": str(row["highway"]),
            "lanes": int(round(float(row["lanes"]))),
            "maxspeed": int(round(float(row["maxspeed"]))),
            "oneway": int(row["oneway"]),
            "start_node_centrality": float(row["start_node_centrality"]),
            "historical_accidents": int(row["historical_accidents"]),
        }

    _score_all_roads.cache_clear()
    print(
        f"Loaded model, SHAP explainer ({shap_explainer is not None}), "
        f"{len(roads_df)} unique road segments, and {mapped_roads_count} exact OSM coordinates."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="Urban Accident Risk Intelligence API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)
    is_weekend: int = Field(..., ge=0, le=1)
    is_rush_hour: int = Field(..., ge=0, le=1)
    length: float = Field(..., gt=0)
    highway: str
    lanes: float = Field(..., gt=0)
    maxspeed: float = Field(..., gt=0)
    oneway: int = Field(..., ge=0, le=1)
    start_node_centrality: float = 0.0


class SimulationRequest(BaseModel):
    baseline: PredictionRequest
    overrides: Dict[str, Any] = Field(default_factory=dict)


@lru_cache(maxsize=256)
def _score_all_roads(hour: int, day_of_week: int, month: int) -> tuple[float, ...]:
    if pipeline is None or roads_df is None:
        raise RuntimeError("Model artifacts are not loaded.")

    batch_df = pd.DataFrame(
        {
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": 1 if day_of_week >= 5 else 0,
            "is_rush_hour": _is_rush_hour(hour, day_of_week),
            "length": roads_df["length"].values,
            "highway": roads_df["highway"].values,
            "lanes": roads_df["lanes"].values,
            "maxspeed": roads_df["maxspeed"].values,
            "oneway": roads_df["oneway"].values,
            "start_node_centrality": roads_df["start_node_centrality"].values,
        }
    )
    probs = pipeline.predict_proba(batch_df[FEATURE_COLS])[:, 1]
    return tuple(float(p) for p in probs)


@app.get("/")
def health_check() -> Dict[str, Any]:
    return {
        "status": "online",
        "model_loaded": pipeline is not None,
        "shap_loaded": shap_explainer is not None,
        "coordinates_loaded": coordinates_loaded,
        "mapped_roads_count": mapped_roads_count,
        "total_roads_indexed": len(roads_df) if roads_df is not None else 0,
        "highway_types": highway_types,
    }


@app.get("/hotspots")
def get_hotspots(
    hour: int = Query(17, ge=0, le=23),
    day_of_week: int = Query(4, ge=0, le=6),
    month: int = Query(10, ge=1, le=12),
    top_n: int = Query(30, ge=1, le=200),
    highway: str = Query("all"),
) -> Dict[str, Any]:
    if pipeline is None or roads_df is None:
        raise HTTPException(status_code=503, detail="Model or road dataset not loaded.")

    probs = np.array(_score_all_roads(hour, day_of_week, month), dtype=float)
    scored = roads_df.copy()
    scored["probability"] = probs
    scored["risk_score"] = np.round(probs * 100.0, 1)

    if highway != "all":
        scored = scored[scored["highway"] == highway]

    if scored.empty:
        return {
            "summary": {
                "evaluated_roads": 0,
                "mean_risk_score": 0.0,
                "critical_count": 0,
                "high_count": 0,
            },
            "hotspots": [],
        }

    scores_arr = scored["risk_score"].values
    summary = {
        "evaluated_roads": int(len(scored)),
        "mean_risk_score": round(float(np.mean(scores_arr)), 1),
        "critical_count": int(np.sum(scores_arr >= 80.0)),
        "high_count": int(np.sum((scores_arr >= 60.0) & (scores_arr < 80.0))),
    }

    top_df = scored.sort_values(
        by=["probability", "historical_accidents"],
        ascending=[False, False],
    ).head(top_n)

    hotspots = []
    for row in top_df.to_dict(orient="records"):
        res = classify_risk(float(row["probability"]))
        lat_val = float(row["lat"]) if pd.notna(row["lat"]) else None
        lon_val = float(row["lon"]) if pd.notna(row["lon"]) else None
        hotspots.append(
            {
                "road_segment_id": str(row["road_segment_id"]),
                "road_name": str(row["road_name"]) if pd.notna(row["road_name"]) else "",
                "lat": lat_val,
                "lon": lon_val,
                "has_exact_coords": bool(row["has_exact_coords"]),
                "highway": str(row["highway"]),
                "length": round(float(row["length"]), 1),
                "lanes": int(round(float(row["lanes"]))),
                "maxspeed": int(round(float(row["maxspeed"]))),
                "oneway": int(row["oneway"]),
                "start_node_centrality": float(row["start_node_centrality"]),
                "historical_accidents": int(row["historical_accidents"]),
                **res,
            }
        )

    return {"summary": summary, "hotspots": hotspots}


@app.get("/timeline/{road_segment_id}")
def get_road_timeline(
    road_segment_id: str,
    day_of_week: int = Query(4, ge=0, le=6),
    month: int = Query(10, ge=1, le=12),
) -> Dict[str, Any]:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    road = roads_lookup.get(road_segment_id)
    if road is None:
        raise HTTPException(
            status_code=404,
            detail=f"Road segment '{road_segment_id}' not found.",
        )

    is_weekend = 1 if day_of_week >= 5 else 0
    rows = [
        {
            "hour": h,
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": is_weekend,
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

    probs = pipeline.predict_proba(pd.DataFrame(rows)[FEATURE_COLS])[:, 1]
    timeline = [{"hour": h, **classify_risk(float(probs[h]))} for h in range(24)]

    return {
        "road_segment_id": road_segment_id,
        "road_name": road["road_name"],
        "day_of_week": day_of_week,
        "month": month,
        "timeline": timeline,
    }


@app.post("/predict")
def predict_risk(request: PredictionRequest) -> Dict[str, Any]:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    input_df = pd.DataFrame([request.model_dump()])[FEATURE_COLS]
    prob = float(pipeline.predict_proba(input_df)[0][1])
    return classify_risk(prob)


@app.post("/explain")
def explain_prediction(request: PredictionRequest) -> Dict[str, Any]:
    if pipeline is None or preprocessor is None or shap_explainer is None:
        raise HTTPException(status_code=503, detail="Model or SHAP explainer not loaded.")

    input_df = pd.DataFrame([request.model_dump()])[FEATURE_COLS]
    prob = float(pipeline.predict_proba(input_df)[0][1])
    pred = classify_risk(prob)

    X_transformed = preprocessor.transform(input_df)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    raw_shap = shap_explainer.shap_values(X_transformed)
    if isinstance(raw_shap, list):
        shap_vals = np.asarray(raw_shap[-1])[0]
    else:
        arr = np.asarray(raw_shap)
        shap_vals = arr[0, :, -1] if arr.ndim == 3 else arr[0]

    expected = shap_explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        base_val = float(np.asarray(expected).ravel()[-1])
    else:
        base_val = float(expected)

    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = [
            _clean_feature_name(name) for name in preprocessor.get_feature_names_out()
        ]
    else:
        feature_names = [f"feature_{i}" for i in range(len(shap_vals))]

    factors = [
        {"feature": name, "shap_value": round(float(val), 4)}
        for name, val in zip(feature_names, shap_vals)
    ]
    factors.sort(key=lambda item: abs(item["shap_value"]), reverse=True)

    return {
        **pred,
        "base_value": round(base_val, 4),
        "top_factors": factors[:7],
    }


@app.post("/simulate")
def simulate_scenario(request: SimulationRequest) -> Dict[str, Any]:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    base_dict = request.baseline.model_dump()
    base_prob = float(pipeline.predict_proba(pd.DataFrame([base_dict])[FEATURE_COLS])[0][1])
    base_result = classify_risk(base_prob)

    sim_dict = dict(base_dict)
    allowed_fields = set(FEATURE_COLS)
    for key, value in request.overrides.items():
        if key in allowed_fields and value is not None:
            sim_dict[key] = value

    if "hour" in request.overrides or "day_of_week" in request.overrides:
        h = int(sim_dict["hour"])
        d = int(sim_dict["day_of_week"])
        sim_dict["is_weekend"] = 1 if d >= 5 else 0
        if "is_rush_hour" not in request.overrides:
            sim_dict["is_rush_hour"] = _is_rush_hour(h, d)

    sim_prob = float(pipeline.predict_proba(pd.DataFrame([sim_dict])[FEATURE_COLS])[0][1])
    sim_result = classify_risk(sim_prob)

    return {
        "baseline": base_result,
        "simulated": sim_result,
        "risk_score_change": round(
            sim_result["risk_score"] - base_result["risk_score"], 1
        ),
    }