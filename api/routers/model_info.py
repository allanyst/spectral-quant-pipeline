from fastapi import APIRouter

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/metrics")
def get_metrics():
    from main import get_model
    artifact = get_model()
    return {
        "n_components": artifact["n_components"],
        "n_samples":    artifact["n_samples"],
        "n_channels":   artifact["n_channels"],
        "elements":     artifact["elements"],
        "metrics":      artifact["metrics"],
    }
