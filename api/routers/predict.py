from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
import numpy as np
import sys, os
sys.path.insert(0, "/app")

router = APIRouter(prefix="/predict", tags=["predict"])

EXPECTED_CHANNELS = 1801


class SpectrumInput(BaseModel):
    intensities: list[float]
    sample_id:   str | None = None

    @field_validator("intensities")
    @classmethod
    def check_length(cls, v):
        if len(v) != EXPECTED_CHANNELS:
            raise ValueError(f"Expected {EXPECTED_CHANNELS} channels, got {len(v)}")
        return v


@router.post("/")
def predict_concentrations(body: SpectrumInput):
    from main import get_model
    artifact = get_model()

    X = np.array(body.intensities, dtype=np.float32).reshape(1, -1)
    X_sc = artifact["scaler"].transform(X)
    Y_pred = artifact["model"].predict(X_sc)[0]
    Y_pred = np.clip(Y_pred, 0, None)

    conc = {el: round(float(v), 6) for el, v in zip(artifact["elements"], Y_pred)}
    total = sum(conc.values())

    return {
        "sample_id":      body.sample_id,
        "concentrations": conc,
        "total":          round(total, 6),
        "n_components":   artifact["n_components"],
    }
