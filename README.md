# Urban Accident Risk Intelligence (Austin, TX)

A full-stack spatio-temporal road safety and accident risk forecasting platform built with **XGBoost**, **SHAP**, **FastAPI**, and **React + Leaflet**. The system scores **14,276** unique Austin road segments by time of day, day of week, and month to surface predicted accident hotspots, 24-hour risk profiles, feature attributions, and interactive What-If scenario simulations.

---

## Key Features

- **Citywide Spatio-Temporal Hotspot Scoring (`/hotspots`)**: Batch-scores all indexed Austin road segments with `@lru_cache` acceleration, road-class filtering (`motorway`, `primary`, `secondary`, `residential`, etc.), and live citywide KPI summaries.
- **24-Hour Risk Forecasting (`/timeline/{road_segment_id}`)**: Runs vectorized 24-hour inference (`00:00–23:00`) for any selected segment to visualize hourly risk curves.
- **Explainable AI via SHAP (`/explain`)**: Computes per-segment `TreeExplainer` feature contributions (`highway`, `length`, `maxspeed`, `hour`, `is_rush_hour`, etc.) to show which inputs push predicted risk up or down.
- **What-If Scenario Simulator (`/simulate`)**: Evaluates model sensitivity to modified speed limits, lane counts, time of day, and traffic regimes against the road's baseline risk.
- **Dark-Mode GIS Operations Dashboard**: Built with React, Vite, and Leaflet, featuring interactive hotspot selection, search, real historical crash counts, and real-time API integration.

---

## Project Structure

```text
.
├── api/
│   └── main.py                         # FastAPI service (scoring, caching, SHAP, simulation)
├── data/
│   └── processed/
│       └── ml_final_features.csv       # Engineered spatio-temporal road features
├── frontend/                           # React + Vite + Leaflet dark-mode dashboard
│   ├── public/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── models/
│   └── xgb_risk_pipeline.pkl           # Serialized preprocessing + XGBoost pipeline
├── .gitignore
├── requirements.txt
└── README.md