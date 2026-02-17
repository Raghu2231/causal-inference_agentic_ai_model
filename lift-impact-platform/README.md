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
2. Review schema/data preview and variable grouping into treatment/covariate/confounder buckets.
3. Run EDA (trends, conversion, missingness, distributions, correlations).
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

**macOS/Linux (bash)**

```bash
cd lift-impact-platform
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.api.main:app --reload --port 8000
```

**Windows PowerShell**

```powershell
cd .\lift-impact-platform
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.api.main:app --reload --port 8000
```

If `Activate.ps1` is missing, create the environment first with `python -m venv .venv`; the activation file is generated under `.venv\Scripts\Activate.ps1` (not in `backend/`).

> **PowerShell note:** `source .venv/bin/activate` is a Bash command and will fail in PowerShell. Use `.\.venv\Scripts\Activate.ps1` instead.

You can also run the helper script:

```powershell
cd .\lift-impact-platform
.\backend\start_backend.ps1
```



### Shell activation quick reference

- **PowerShell**: `.\.venv\Scripts\Activate.ps1`
- **CMD**: `.\.venv\Scripts\activate.bat`
- **Git Bash / WSL / Linux / macOS**: `source .venv/bin/activate`

If `npm` is not recognized in PowerShell, install Node.js LTS and reopen terminal so PATH refreshes.

### 2) Frontend (React + Node.js)

**macOS/Linux (bash)**

```bash
cd lift-impact-platform/frontend
npm install
npm run dev
```

**Windows PowerShell**

```powershell
cd .\lift-impact-platform\frontend
npm install
npm run dev
```

Or run:

```powershell
cd .\lift-impact-platform
.\frontend\start_frontend.ps1
```

Optional frontend API override and timeout tuning:

```bash
# frontend/.env
VITE_API_BASE_URL=http://localhost:8000
VITE_API_TIMEOUT_MS=120000
VITE_UPLOAD_TIMEOUT_MS=180000
VITE_EDA_TIMEOUT_MS=300000
VITE_API_RETRY_COUNT=3
VITE_API_RETRY_DELAY_MS=1200
```

The upload page includes backend health status, drag-and-drop upload, and a progress bar for large Excel files.

For large datasets, EDA is loaded separately via **Load EDA Summary** after upload to avoid upload-step timeouts.

EDA includes per-variable statistics (mean/std/quartiles) and a feature-engineering diagnostic view (lags/rolling/adstock summaries) for model-readiness checks.

If the UI shows `Request timed out while connecting to http://localhost:8000`, verify backend availability first:

```bash
curl http://localhost:8000/health
```

Expected response: `{"status":"ok"}`.

If connection errors happen repeatedly after selecting a file, backend may be restarting/crashing while processing the file. Check the backend terminal for Python traceback and verify dependencies are installed in the same venv used by uvicorn.

## Deployment

- Deploy FastAPI backend to containers/serverless runtime.
- Build frontend with `npm run build` and serve static assets (Nginx/CDN).
- Persist `artifacts/` to cloud object storage for multi-instance operation.

## API overview

- `POST /upload` — upload Excel and detect schema
- `GET /preview/{file_id}` — schema preview + variable partition + sample rows
- `GET /eda/{file_id}` — EDA outputs
- `POST /run/{file_id}` — run Path A + Path B and return lift summary
- `POST /insights/{file_id}` — generate automated narrative insights from model outputs (LLM-backed with fallback)

## Notes

- Models are end-to-end runnable and persist artifacts.
- Supports scenario multiplier and channel isolation.
- Includes automated pipeline test scaffold in `tests/test_pipeline.py`.


## Optional LLM integration for automated insights

Set these backend environment variables to enable LLM-generated narrative insights:

```bash
LLM_API_URL=<chat-completions-endpoint>
LLM_API_KEY=<api-key-if-required>
LLM_MODEL=gpt-4o-mini
```

If these are not configured, the platform automatically returns rule-based insights derived from Path A/Path B lift metrics.
