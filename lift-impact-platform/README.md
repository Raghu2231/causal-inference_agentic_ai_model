# Lift Impact Platform

A production-style causal lift platform that re-implements the source repository's core ideas: a two-path causal chain from **Suggestions → Actions → Outcomes (TRX/NBRX)**.

## Path A vs Path B (Conceptual)

- **Path A: Incremental Action Lift from Suggestions**
  - Goal: estimate how many additional rep actions are caused by suggestions above what would have happened without suggestion exposure.
  - We model two components:
    1. **Propensity model** for action uptake under suggestion exposure and confounders.
    2. **Treatment-effect model** (T-learner) for action count uplift from suggestions.
  - Outputs: incremental actions by row, by channel, plus aggregate lift.

- **Path B: Incremental Outcome Lift from Actions**
  - Goal: estimate additional TRX/NBRX caused by actions above no-action baseline.
  - Uses treatment-effect outcome models with controls for historical outcome levels, seasonality, and rep/HCP behavior profiles.
  - Outputs: incremental TRX and NBRX, action-type/channel-attributable effects.

## Mathematical intuition (plain language)

- For each observation, we infer two potential outcomes:
  - what would happen **with treatment** (suggestion/action)
  - what would happen **without treatment**
- The difference is individual uplift (ITE). Aggregating ITE gives incremental lift.
- Path chaining is:
  1. `incremental_actions = E[action | do(suggestion=1)] - E[action | do(suggestion=0)]`
  2. `incremental_trx/nbrx = E[outcome | do(action=1)] - E[outcome | do(action=0)]`
- Path B consumes Path A incremental-action outputs directly for chained estimates and scenario simulations.

## Assumptions & limitations

- Observed confounders are sufficiently captured by engineered covariates.
- Positivity/overlap: treated and control groups both exist across feature space.
- Temporal causality is approximated via lagged and rolling windows.
- T-learner is flexible but not fully robust to severe hidden confounding.
- Automatic schema detection relies on naming conventions for column inference.

## How lift is chained

1. Upload Excel and auto-detect suggestions/actions/outcomes/time/IDs.
2. Build transformation layer (lags, windows, indicators, normalization, one-hot).
3. Run Path A propensity + treatment effect → incremental actions.
4. Feed incremental action signal into Path B outcome-lift model.
5. Produce aggregate + channel + rep + HCP incremental metrics.

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
│   ├── src/
│   │   ├── pages/
│   │   ├── components/
│   │   └── services/
│   ├── package.json
│   └── vite.config.js
├── data_contracts/
├── tests/
├── README.md
└── requirements.txt
```

## Run locally

### 1) Backend (FastAPI)

```bash
cd lift-impact-platform
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.api.main:app --reload --port 8000
```

### 2) Frontend (React + Vite)

```bash
cd lift-impact-platform/frontend
npm install
npm run dev
```

By default, the frontend calls `http://localhost:8000`. To use another backend URL:

```bash
# in frontend/.env
VITE_API_BASE_URL=http://localhost:8000
```

## Deployment

- Backend can run in Kubernetes, ECS, Cloud Run, or VM containers.
- Frontend can be built as static assets (`npm run build`) and served via CDN/Nginx.
- Persist artifacts (`artifacts/`) to object storage (S3/GCS/Azure Blob) for multi-instance stateless operation.

## API overview

- `POST /upload` — upload Excel, schema detection
- `GET /eda/{file_id}` — generated EDA package
- `POST /run/{file_id}` — execute Path A + Path B and return summary

## Notes on implementation quality

- No placeholder model stubs: each model is fit/predict ready.
- Intermediate artifacts are persisted (`parquet`, `csv`, `json`).
- Scenario simulation and channel isolation are supported in lift engine.

## Troubleshooting

- If upload fails with connection-refused, ensure backend is running on port `8000`.
- For browser CORS errors, verify backend includes CORS middleware and restart `uvicorn`.
