# Spectral Quantification Pipeline

Multi-element XRF quantification pipeline using PLS2 regression over a Bronze→Silver→Gold medallion architecture. Ingests raw WDXRF spectra stored as PostgreSQL `FLOAT[]` arrays, transforms them with dbt, trains a calibration model, and serves predictions via REST API.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Synthetic WDXRF spectra (300 samples · 1801 channels · 0.05°) │
└────────────────────────────┬────────────────────────────────────┘
                             │  psycopg2 + execute_values
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  BRONZE  PostgreSQL 15                                          │
│  bronze.xrf_spectra                                             │
│    spectrum_id  SERIAL                                          │
│    two_theta    FLOAT[]   ← 2θ grid, 1801 values               │
│    intensities  FLOAT[]   ← counts per channel                  │
│    conc_na/mg/al/fe/si/p  FLOAT  ← true concentrations         │
└────────────────────────────┬────────────────────────────────────┘
                             │  dbt-postgres · UNNEST WITH ORDINALITY
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  SILVER  (dbt views)                                            │
│  silver.stg_xrf_spectra                                         │
│    one row per (spectrum × channel)                             │
│    channel_idx · two_theta · intensity  ← array exploded        │
└────────────────────────────┬────────────────────────────────────┘
                             │  dbt · window aggregation
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  GOLD  (dbt tables)                                             │
│  gold.fact_spectral_peaks    ← net peak intensities per element │
│  gold.fact_sample_summary    ← features + labels for PLS        │
└────────────────────────────┬────────────────────────────────────┘
                             │  scikit-learn PLSRegression
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  PLS2 MODEL  (models/pls_model.pkl)                             │
│  6 elements simultaneously · n_components=15 · R²>0.999        │
└────────────────────────────┬────────────────────────────────────┘
                             │  FastAPI
                             ▼
              POST /predict  →  { Na: 0.31, Mg: 0.22, … }
```

## Stack

| Layer | Technology |
|---|---|
| Orchestration | Apache Airflow 2.8.1 |
| Storage | PostgreSQL 15 (`FLOAT[]` arrays) |
| Transformation | dbt-postgres 1.8 |
| ML | scikit-learn PLSRegression (PLS2) |
| API | FastAPI + uvicorn |
| Dashboard | Streamlit + Plotly |
| Containers | Docker Compose (6 services) |

## Key Technical Decisions

**`FLOAT[]` arrays in PostgreSQL** — Raw spectra are stored as native array columns instead of one-row-per-channel. This keeps the Bronze table at 300 rows regardless of spectral resolution, and makes bulk ingestion trivial via `psycopg2.extras.execute_values`. The wide→long conversion happens once in Silver via `UNNEST ... WITH ORDINALITY`.

**PLS2 over peak-picking** — Rather than manually extracting peak heights, the PLS model receives the full 1801-channel spectrum as input. Latent variables capture correlated inter-element variance automatically, including matrix absorption effects that would be missed by simple univariate calibration.

**dbt `UNNEST` in Silver** — The key transformation is:
```sql
FROM bronze.xrf_spectra,
     LATERAL UNNEST(two_theta, intensities)
         WITH ORDINALITY AS t(two_theta_val, intensity_val, channel_idx)
```
This converts the wide array format into a proper analytical relation without any Python preprocessing step, keeping the transformation logic declarative and testable.

## Elements and Spectral Model

| Element | Kα 2θ (°) | Sensitivity (counts/wt-frac) |
|---------|-----------|------------------------------|
| Na | 55.1 | 12 000 |
| Mg | 45.2 | 18 000 |
| Al | 57.5 | 22 000 |
| Fe | 17.5 | 45 000 |
| Si | 75.0 | 30 000 |
| P  | 89.3 |  9 500 |

Spectra include Poisson counting noise and inter-element absorption (matrix effects) modelled via an empirical correction matrix. Mix types: unary (60), binary (120), ternary (120).

## Model Performance

| Element | R² | RMSE (wt-frac) |
|---------|----|----------------|
| Na | 0.9994 | 0.0056 |
| Mg | 0.9998 | 0.0035 |
| Al | 0.9996 | 0.0046 |
| Fe | 0.9998 | 0.0031 |
| Si | 0.9995 | 0.0055 |
| P  | 0.9996 | 0.0051 |

n_components = 15, selected by 5-fold cross-validation (RMSECV).

## Airflow DAG

```
ingest_bronze >> dbt_run >> dbt_test >> train_pls
```

Schedule: `0 7 * * 1-5` (weekdays 07:00 UTC). All tasks have `retries=2`.

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/predict` | Submit spectrum → concentrations |
| `GET` | `/model/metrics` | R², RMSE, n_components |
| `GET` | `/samples` | Browse Gold layer |
| `GET` | `/samples/{id}` | Single sample with raw spectrum |

**Predict example:**
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"sample_id": "T-NaAlFe-001", "intensities": [...]}'
```

Response:
```json
{
  "sample_id": "T-NaAlFe-001",
  "concentrations": {"na": 0.312, "mg": 0.0, "al": 0.441, "fe": 0.247, "si": 0.0, "p": 0.0},
  "total": 1.000,
  "n_components": 15
}
```

## Setup

**Prerequisites:** Docker, Docker Compose, Anaconda/Miniconda

```bash
git clone git@gitlab.com:plasmallan/spectral-quant-pipeline.git
cd spectral-quant-pipeline

# Create local environment (used for ingestion and model training scripts)
conda env create -f environment.yml
conda activate spectral-quant

# Start PostgreSQL and dbt
docker compose up -d postgres dbt
sleep 15

# Generate synthetic spectra and load Bronze
python src/ingest_bronze.py

# Run Silver and Gold transformations
docker exec spectral-quant-pipeline-dbt-1 dbt run --profiles-dir /root/.dbt

# Train PLS model
python models/pls_calibration.py

# Start full stack
docker compose up -d
```

Services:
- Airflow UI: http://localhost:8082 (admin / admin)
- API docs:   http://localhost:8001/docs
- Dashboard:  http://localhost:8502

## Project Structure

```
.
├── airflow/
│   ├── Dockerfile                  # airflow + psycopg2 + scikit-learn
│   └── dags/spectral_pipeline.py   # ingest >> dbt >> train_pls
├── api/
│   ├── main.py
│   └── routers/
│       ├── predict.py              # POST /predict
│       ├── samples.py              # GET /samples
│       └── model_info.py           # GET /model/metrics
├── dashboard/app.py                # Streamlit spectrum viewer
├── dbt/
│   ├── models/
│   │   ├── staging/stg_xrf_spectra.sql      # UNNEST arrays
│   │   └── marts/
│   │       ├── fact_spectral_peaks.sql       # net peak intensities
│   │       └── fact_sample_summary.sql       # PLS feature table
│   └── macros/generate_schema_name.sql
├── docker/init-db/
│   ├── 01_schemas.sql              # bronze / silver / gold schemas
│   └── 02_bronze.sql               # xrf_spectra with FLOAT[] columns
├── models/pls_calibration.py       # PLS2 train + inference
└── src/
    ├── generate/synthetic.py       # WDXRF spectrum simulator
    └── ingestion/loaders.py        # psycopg2 bulk loader
```
