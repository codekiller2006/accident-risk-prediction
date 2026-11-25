# Urban Accident Risk Intelligence (Austin, TX)

A full-stack spatio-temporal road safety and accident risk forecasting platform built with **XGBoost**, **SHAP**, **FastAPI**, and **React + Leaflet**. The system scores **14,276** unique Austin road segments by hour, day of week, and month to surface predicted accident hotspots on real OpenStreetMap street midpoints, generate 24-hour risk forecasts, explain predictions in plain English via SHAP, and simulate What-If road and traffic interventions.

---

## Key Features

- **Citywide Spatio-Temporal Hotspot Scoring (`/hotspots`)**: Vectorized batch inference across all 14,276 indexed Austin road segments with `@lru_cache` acceleration, road-class filtering (`motorway`, `primary`, `secondary`, `residential`, etc.), and citywide KPI summaries.
- **Real OSM Street Mapping**: Plots hotspots on OpenStreetMap road midpoints (`data/processed/road_coordinates.csv`) with street names, auto-fitted map bounds, and dark-mode street tiles requiring no paid map API keys.
- **24-Hour Risk Forecasting (`/timeline/{road_segment_id}`)**: Computes a vectorized 24-hour (`00:00–23:00`) risk profile for any selected road segment alongside road class, segment length, one-way status, and historical crash counts.
- **Explainable AI via SHAP (`/explain`)**: Computes exact per-segment `TreeExplainer` log-odds attributions and translates top risk-raising and risk-reducing drivers into plain-English summaries on the dashboard.
- **What-If Scenario Simulator (`/simulate`)**: Evaluates model sensitivity under modified speed limits, lane counts, time of day, and traffic regimes against the segment's baseline risk.

---

## Dataset & Preprocessing Pipeline

- **Data Sources**: Austin, Texas crash records from the **US Accidents (2016–2023)** dataset map-matched onto Austin's drivable road network extracted via **OpenStreetMap (OSMnx)**.
- **Coverage**: `274,128` spatio-temporal records across `14,276` unique road segments spanning **`2016-06-14` to `2023-03-31`**.
- **Negative Sampling (`1 : 2.00`)**: `91,376` historical crash events (`Target = 1`) paired with `182,752` negative non-crash samples (`Target = 0`) to keep training tractable and balanced without generating hundreds of millions of zero-event rows.
- **Cleaning & Leakage-Free Preprocessing**: The processed feature table contains **0 missing values** (`length`: `2.1–9,702 m`, `lanes`: `1–7`, `maxspeed`: `15–80 mph`). For safe inference on unseen inputs, `xgb_risk_pipeline.pkl` embeds a scikit-learn `ColumnTransformer`:
  - **Numeric features** (`hour`, `day_of_week`, `month`, `is_weekend`, `is_rush_hour`, `length`, `lanes`, `maxspeed`, `oneway`, `start_node_centrality`): `SimpleImputer(strategy="median")` + `StandardScaler()`.
  - **Categorical feature** (`highway`): `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`.

---

## Model Comparison & Production Pipeline

To prevent temporal data leakage, models are evaluated on a strict **chronological 80/20 holdout split** (`219,302` training rows up to `2021-09-17`; `54,826` unseen future test rows from `2021-09-17` onward) using `audit_and_benchmark.py`:

| Model | ROC-AUC | PR-AUC | F1-Score | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7164 | 0.4944 | 0.4904 | 0.3910 | 0.6578 |
| **Random Forest** | 0.7597 | 0.5476 | 0.5256 | 0.4259 | **0.6864** |
| **XGBoost (Selected)** | **0.7791** | **0.5789** | **0.5447** | **0.4526** | 0.6839 |

- **Why XGBoost**: Captures non-linear interactions between road geometry and temporal traffic proxies better than linear baselines (`+6.27%` ROC-AUC and `+8.45%` PR-AUC over Logistic Regression), outperforms Random Forest on ROC-AUC, PR-AUC, F1, and Precision, runs fast CPU batch inference across all 14,276 roads, and supports exact polynomial-time SHAP explanations via `TreeExplainer`.
- **Deployed Pipeline (`models/xgb_risk_pipeline.pkl`)**: `XGBClassifier` configured with `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `eval_metric="aucpr"` (PR-AUC optimization for imbalanced risk ranking), and `scale_pos_weight=1.816`.

---

## Project Structure

```text
.
├── api/
│   └── main.py                         # FastAPI inference service (scoring, caching, SHAP, simulation)
├── data/
│   └── processed/
│       ├── ml_final_features.csv       # Engineered spatio-temporal road features (274,128 rows)
│       └── road_coordinates.csv        # OpenStreetMap segment midpoints & street names
├── frontend/                           # React + Vite + Leaflet dark-mode GIS dashboard
│   ├── public/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── models/
│   └── xgb_risk_pipeline.pkl           # Serialized ColumnTransformer + XGBoost pipeline
├── audit_and_benchmark.py              # Dataset audit & chronological 3-model benchmark script
├── build_road_coordinates.py           # Offline OSMnx script to generate road_coordinates.csv
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Local Setup & Running

### 1. Prerequisites
- **Python 3.10+**
- **Node.js 18+** & `npm`
- **macOS OpenMP Runtime** (required by XGBoost):
  ```bash
  brew install libomp
  ```

### 2. Start the FastAPI Backend
From the project root directory:
```bash
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```
Verify the backend at `http://127.0.0.1:8000/` (Swagger UI at `http://127.0.0.1:8000/docs`).

### 3. Start the React Frontend
Open a second terminal:
```bash
cd frontend
npm install
npm run dev
```
Open **`http://localhost:5173`** in your browser.

### 4. Reproducing Benchmarks & Coordinates (Optional)
```bash
# Reproduce dataset audit and chronological 3-model benchmark
python audit_and_benchmark.py

# Regenerate OSM road midpoints & street names (requires: pip install osmnx)
python build_road_coordinates.py
```

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Health check, loaded model/SHAP/coordinate status, indexed road count, and road classes |
| `GET` | `/hotspots` | Top-N ranked hotspots (with OSM coordinates & street names) and citywide KPI summary |
| `GET` | `/timeline/{road_segment_id}` | Vectorized 24-hour (`0–23`) risk forecast for a specific road segment |
| `POST` | `/predict` | Single-segment risk probability, percentage score, and risk tier |
| `POST` | `/explain` | Prediction plus top-7 SHAP log-odds feature attributions |
| `POST` | `/simulate` | Baseline vs. modified scenario risk comparison |

---

## Technical Scope & Known Limitations

- **Relative Risk Score vs. Calibrated Probability**: Because the model is trained with `1:2` negative sampling and `scale_pos_weight=1.816`, a `90%+` score represents extreme **relative** crash risk compared to other Austin road-hours—not a literal 90% per-vehicle crash probability.
- **Correlation vs. Causality**: SHAP values and What-If simulations reflect how the trained XGBoost model weights correlated historical patterns, not causal proof that altering a physical road sign or lane count will reduce real-world crashes by the exact simulated delta.
- **Zero-Variance Centrality Column**: In the current processed dataset, `start_node_centrality` is `0` across all rows, so XGBoost assigns it zero weight; computing non-zero betweenness centrality on the Austin road graph is left as a future feature upgrade.
- **Reporting & Exposure Bias**: Public traffic-API crash logs capture highway and arterial incidents more consistently than minor residential crashes, and predictions rely on temporal proxies (`hour`, `is_rush_hour`, `month`) rather than live traffic volume or real-time weather feeds.