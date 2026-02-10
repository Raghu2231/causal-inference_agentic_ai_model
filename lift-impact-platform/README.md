# Lift Impact Platform

A production-style causal lift platform that re-implements the source repository's core logic using a two-path causal chain from **Suggestions → Actions → Outcomes (TRX/NBRX)**.

## Path A vs Path B (Conceptual)

- **Path A: Incremental Action Lift from Suggestions**
  - Measures additional rep actions caused by suggestions above baseline behavior.
  - Includes:
    1. **Propensity Model** to estimate action probability under suggestion exposure and confounders.
    2. **Treatment Effect Model** to estimate incremental action volume attributable to suggestions.
  - Produces channel lift, rep/HCP lift, row-level and aggregate incremental actions.

- **Path B: Incremental Outcome Lift from Actions**
  - Measures additional TRX/NBRX caused by observed or incremental actions from Path A.
  - Uses uplift-style treated/control regressors with prescribing baseline and behavior controls.
  - Produces incremental TRX, incremental NBRX, and aggregate totals.

## Mathematical intuition (plain language)

Each record has two potential states: treated and untreated. The model estimates both expected outcomes and subtracts them to compute uplift.

- Path A uplift: expected actions with suggestions minus expected actions without suggestions.
- Path B uplift: expected TRX/NBRX with actions minus expected TRX/NBRX without actions.

Chaining Path A to Path B translates suggestion-driven behavior lift into outcome lift.

## Assumptions & limitations

- Confounders are assumed mostly observed in engineered features.
- Treated and control overlap is required for stable uplift estimates.
- Temporal causality is approximated with lags/rolling windows.
- Unobserved confounding can bias treatment effects.
- Auto-schema detection depends on column naming patterns.

## How lift is chained

1. Upload Excel and detect suggestions, actions, outcomes, time, HCP ID, Rep ID.
2. Run EDA (trends, conversion, missingness, distributions, correlations).
3. Build transformation layer (indicators, lags/windows, normalization, one-hot encoding).
4. Run Path A propensity + treatment effect models to estimate incremental actions.
5. Run Path B outcome-lift model using incremental actions and controls.
6. Return and persist aggregate/channel/rep/HCP outputs.

## Repository structure

```text
lift-impact-platform/
├── backend/
│   ├── api/
│   ├── services/
│   ├── models/
│   │   ├── path_a/
│   │   └── path_b/
│   ├── eda/
│   ├── transformations/
│   └── utils/
├── frontend/
│   ├── pages/
│   ├── components/
│   ├── services/
│   ├── App.jsx
│   ├── main.jsx
│   ├── styles.css
│   ├── package.json
│   └── vite.config.js
├── data_contracts/
├── tests/
├── README.md
└── requirements.txt
```

## Web application

- **Backend**: FastAPI (`/upload`, `/eda/{file_id}`, `/run/{file_id}`)
- **Frontend**: React + Node.js (Vite)
- **Charts**: Plotly
- **State**: Stateless API calls with file_id session token

Pages implemented:
- Upload Page
- EDA Dashboard
- Path A Dashboard
- Path B Dashboard
- Final Lift Summary + downloadable JSON reports

## Run locally

### 1) Backend

```bash
cd lift-impact-platform
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.api.main:app --reload --port 8000
```

### 2) Frontend (React + Node.js)

```bash
cd lift-impact-platform/frontend
npm install
npm run dev
```

Optional frontend API override:

```bash
# frontend/.env
VITE_API_BASE_URL=http://localhost:8000
```

## Deployment

- Deploy FastAPI backend to containers/serverless runtime.
- Build frontend with `npm run build` and serve static assets (Nginx/CDN).
- Persist `artifacts/` to cloud object storage for multi-instance operation.

## API overview

- `POST /upload` — upload Excel and detect schema
- `GET /eda/{file_id}` — EDA outputs
- `POST /run/{file_id}` — run Path A + Path B and return lift summary

## Notes

- Models are end-to-end runnable and persist artifacts.
- Supports scenario multiplier and channel isolation.
- Includes automated pipeline test scaffold in `tests/test_pipeline.py`.
