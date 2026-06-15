"""
PLS2 calibration model for multi-element XRF quantification.

Strategy:
- Features: full spectrum (1801 channels) from bronze.xrf_spectra
  PLS works best on the raw spectral matrix — no manual peak picking needed.
  The latent variables capture correlated variance across all channels.
- Labels: 6 concentrations (Na, Mg, Al, Fe, Si, P) from gold.fact_sample_summary
- Model: PLSRegression (PLS2 — one model, all elements simultaneously)
- n_components chosen by 5-fold cross-validation (RMSECV)
- Artifact saved to MODEL_PATH as a pickle containing model + metadata
"""
from __future__ import annotations
import os
import pickle
import numpy as np
import psycopg2
from pathlib import Path
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

DB_URL    = os.environ.get("XRF_DB_URL",   "postgresql://xrf:xrf_pass@localhost:5434/spectra_db")
MODEL_PATH = Path(os.environ.get("MODEL_PATH", Path(__file__).parent / "pls_model.pkl"))

ELEMENTS  = ["na", "mg", "al", "fe", "si", "p"]
MAX_COMPONENTS = 15


def load_training_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns:
        X : (n_samples, n_channels) — raw spectra (float32)
        Y : (n_samples, 6)          — concentrations
        sample_ids : list[str]
    """
    conn = psycopg2.connect(DB_URL)
    try:
        with conn.cursor() as cur:
            # Fetch spectra ordered the same as summary for row alignment
            cur.execute("""
                SELECT s.sample_id,
                       s.intensities,
                       gs.conc_na, gs.conc_mg, gs.conc_al,
                       gs.conc_fe, gs.conc_si, gs.conc_p
                FROM bronze.xrf_spectra s
                JOIN gold.fact_sample_summary gs USING (sample_id)
                ORDER BY s.sample_id
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    sample_ids = [r[0] for r in rows]
    X = np.array([r[1]  for r in rows], dtype=np.float32)
    Y = np.array([r[2:] for r in rows], dtype=np.float32)
    return X, Y, sample_ids


def select_n_components(X: np.ndarray, Y: np.ndarray) -> int:
    """Choose n_components via 5-fold CV minimizing mean RMSECV across all elements."""
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_n, best_rmsecv = 1, np.inf
    for n in range(1, MAX_COMPONENTS + 1):
        pls = PLSRegression(n_components=n, scale=False)
        Y_cv = cross_val_predict(pls, X_sc, Y, cv=kf)
        rmsecv = np.mean([
            np.sqrt(mean_squared_error(Y[:, i], Y_cv[:, i]))
            for i in range(Y.shape[1])
        ])
        if rmsecv < best_rmsecv:
            best_rmsecv, best_n = rmsecv, n
    return best_n


def train_and_save() -> dict:
    print("Loading training data from PostgreSQL...")
    X, Y, sample_ids = load_training_data()
    print(f"  X: {X.shape}  Y: {Y.shape}")

    print("Selecting optimal n_components via 5-fold CV...")
    n_comp = select_n_components(X, Y)
    print(f"  Best n_components: {n_comp}")

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    pls = PLSRegression(n_components=n_comp, scale=False)
    pls.fit(X_sc, Y)

    Y_pred = pls.predict(X_sc)
    Y_pred = np.clip(Y_pred, 0, None)

    metrics = {}
    for i, el in enumerate(ELEMENTS):
        r2   = r2_score(Y[:, i], Y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(Y[:, i], Y_pred[:, i]))
        metrics[el] = {"r2": round(float(r2), 4), "rmse": round(float(rmse), 6)}
        print(f"  {el.upper():2s}  R²={r2:.4f}  RMSE={rmse:.6f}")

    artifact = {
        "model":        pls,
        "scaler":       scaler,
        "n_components": n_comp,
        "n_samples":    len(sample_ids),
        "n_channels":   X.shape[1],
        "elements":     ELEMENTS,
        "metrics":      metrics,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Model saved to {MODEL_PATH}")
    return metrics


# ── Inference helper (used by API) ────────────────────────────────────────────

def load_model() -> dict:
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(intensities: list[float], artifact: dict | None = None) -> dict[str, float]:
    """Return predicted concentrations for a single spectrum."""
    if artifact is None:
        artifact = load_model()
    X = np.array(intensities, dtype=np.float32).reshape(1, -1)
    X_sc = artifact["scaler"].transform(X)
    Y_pred = artifact["model"].predict(X_sc)[0]
    Y_pred = np.clip(Y_pred, 0, None)
    return {el: round(float(v), 6) for el, v in zip(artifact["elements"], Y_pred)}


if __name__ == "__main__":
    train_and_save()
