from fastapi import FastAPI
import os, pickle
from pathlib import Path

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/app/models/pls_model.pkl"))
DB_URL     = os.environ.get("XRF_DB_URL", "postgresql://xrf:xrf_pass@postgres:5432/spectra_db")

# Lazy-load model artifact (loaded once on first request)
_artifact: dict | None = None

def get_model() -> dict:
    global _artifact
    if _artifact is None:
        with open(MODEL_PATH, "rb") as f:
            _artifact = pickle.load(f)
    return _artifact

app = FastAPI(
    title="Spectral Quantification API",
    description="PLS2 multi-element XRF quantification — Bronze→Silver→Gold pipeline",
    version="1.0.0",
)

from routers import predict, samples, model_info
app.include_router(predict.router)
app.include_router(samples.router)
app.include_router(model_info.router)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _artifact is not None}
