"""
Synthetic WDXRF spectra generator.

Models intensity as a sum of Gaussian peaks (one per element Kα line)
with concentration-proportional amplitude, plus background and Poisson noise.
Matrix effects (absorption/enhancement) are approximated with a simple
inter-element correction term — realistic enough for PLS calibration demos.

Elements and approximate Kα 2θ positions (PET analyzing crystal):
    Na  55.1°   Mg  45.2°   Al  57.5°
    Si  75.0°   P   89.3°   Fe  17.5°  (LiF 200 crystal)

For simplicity we put all on a single 2θ grid 10°–100° at 0.05° step (1801 channels).
"""
from __future__ import annotations
import numpy as np
import itertools
from dataclasses import dataclass, field

# ── Instrument grid ────────────────────────────────────────────────────────────
TWO_THETA = np.arange(10.0, 100.05, 0.05)   # 1801 channels
N_CHANNELS = len(TWO_THETA)

# ── Element definitions ────────────────────────────────────────────────────────
@dataclass
class Element:
    symbol:     str
    peak_2t:    float   # 2θ of Kα line, degrees
    sigma:      float   # peak width (FWHM / 2.355), degrees
    sensitivity: float  # counts per unit mass fraction at 1.0 conc

ELEMENTS: dict[str, Element] = {
    "Na": Element("Na", 55.1, 0.18, 12_000),
    "Mg": Element("Mg", 45.2, 0.15, 18_000),
    "Al": Element("Al", 57.5, 0.16, 22_000),
    "Fe": Element("Fe", 17.5, 0.12, 45_000),
    "Si": Element("Si", 75.0, 0.20, 30_000),
    "P":  Element("P",  89.3, 0.22,  9_500),
}
ELEMENT_KEYS = list(ELEMENTS.keys())

# Simple inter-element absorption coefficients (row absorbs column's signal)
# Values exaggerated slightly to make matrix effects visible in PLS loading plots
MATRIX_ALPHA = np.array([
#   Na    Mg    Al    Fe    Si    P
    [0.00, 0.04, 0.03, 0.08, 0.02, 0.01],  # Na absorbed by
    [0.03, 0.00, 0.05, 0.10, 0.03, 0.02],  # Mg
    [0.02, 0.04, 0.00, 0.12, 0.04, 0.01],  # Al
    [0.01, 0.02, 0.03, 0.00, 0.05, 0.02],  # Fe
    [0.02, 0.03, 0.04, 0.06, 0.00, 0.01],  # Si
    [0.01, 0.01, 0.02, 0.05, 0.02, 0.00],  # P
], dtype=float)

BACKGROUND_LEVEL = 800.0   # baseline counts (bremsstrahlung)
BACKGROUND_SLOPE = -3.0    # slight downward slope across 2θ range


def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def simulate_spectrum(concentrations: dict[str, float], rng: np.random.Generator) -> np.ndarray:
    """Return intensity array (counts) for given element concentrations."""
    conc_vec = np.array([concentrations.get(el, 0.0) for el in ELEMENT_KEYS])

    # Background
    bg = BACKGROUND_LEVEL + BACKGROUND_SLOPE * (TWO_THETA - TWO_THETA[0])

    spectrum = bg.copy()
    for i, el_key in enumerate(ELEMENT_KEYS):
        el = ELEMENTS[el_key]
        ci = conc_vec[i]
        if ci <= 0:
            continue
        # Matrix correction: absorption by all other elements
        absorption = 1.0 - MATRIX_ALPHA[i] @ conc_vec
        net_intensity = el.sensitivity * ci * max(absorption, 0.3)
        spectrum += net_intensity * gaussian(TWO_THETA, el.peak_2t, el.sigma)

    # Poisson noise (sqrt of signal)
    spectrum = rng.poisson(np.maximum(spectrum, 1).astype(int)).astype(float)
    return spectrum


# ── Concentration sampling ─────────────────────────────────────────────────────

def _dirichlet_conc(elements: list[str], rng: np.random.Generator) -> dict[str, float]:
    """Sample concentrations that sum to 1 via Dirichlet."""
    alpha = np.ones(len(elements)) * 2.0  # slightly pushed toward equal mix
    raw = rng.dirichlet(alpha)
    return {el: float(v) for el, v in zip(elements, raw)}


def generate_samples(
    n_unary:   int = 60,
    n_binary:  int = 120,
    n_ternary: int = 120,
    seed:      int = 42,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    unary_elements = ELEMENT_KEYS
    binary_pairs   = list(itertools.combinations(ELEMENT_KEYS, 2))
    ternary_triples = list(itertools.combinations(ELEMENT_KEYS, 3))

    # ── Unary ──────────────────────────────────────────────────────────────────
    per_element = max(1, n_unary // len(unary_elements))
    for el in unary_elements:
        for rep in range(per_element):
            c = float(rng.uniform(0.05, 1.0))
            conc = {e: 0.0 for e in ELEMENT_KEYS}
            conc[el] = c
            sample_id = f"U-{el}-{rep+1:03d}"
            records.append(_build_record(sample_id, "unary", conc, rng))

    # ── Binary ─────────────────────────────────────────────────────────────────
    per_pair = max(1, n_binary // len(binary_pairs))
    for (e1, e2) in binary_pairs:
        for rep in range(per_pair):
            raw = _dirichlet_conc([e1, e2], rng)
            conc = {e: 0.0 for e in ELEMENT_KEYS}
            conc[e1] = raw[e1]
            conc[e2] = raw[e2]
            sample_id = f"B-{e1}{e2}-{rep+1:03d}"
            records.append(_build_record(sample_id, "binary", conc, rng))

    # ── Ternary ────────────────────────────────────────────────────────────────
    per_triple = max(1, n_ternary // len(ternary_triples))
    for (e1, e2, e3) in ternary_triples:
        for rep in range(per_triple):
            raw = _dirichlet_conc([e1, e2, e3], rng)
            conc = {e: 0.0 for e in ELEMENT_KEYS}
            conc[e1] = raw[e1]
            conc[e2] = raw[e2]
            conc[e3] = raw[e3]
            sample_id = f"T-{e1}{e2}{e3}-{rep+1:03d}"
            records.append(_build_record(sample_id, "ternary", conc, rng))

    return records


def _build_record(sample_id: str, mix_type: str, conc: dict[str, float], rng) -> dict:
    spectrum = simulate_spectrum(conc, rng)
    return {
        "sample_id":    sample_id,
        "mix_type":     mix_type,
        "conc_na":      conc.get("Na", 0.0),
        "conc_mg":      conc.get("Mg", 0.0),
        "conc_al":      conc.get("Al", 0.0),
        "conc_fe":      conc.get("Fe", 0.0),
        "conc_si":      conc.get("Si", 0.0),
        "conc_p":       conc.get("P",  0.0),
        "two_theta":    TWO_THETA.tolist(),
        "intensities":  spectrum.tolist(),
        "n_channels":   N_CHANNELS,
    }


if __name__ == "__main__":
    samples = generate_samples()
    print(f"Generated {len(samples)} spectra  |  {N_CHANNELS} channels each")
    print(f"  Unary:   {sum(1 for s in samples if s['mix_type'] == 'unary')}")
    print(f"  Binary:  {sum(1 for s in samples if s['mix_type'] == 'binary')}")
    print(f"  Ternary: {sum(1 for s in samples if s['mix_type'] == 'ternary')}")
    print(f"  2θ range: {TWO_THETA[0]}° – {TWO_THETA[-1]}°  step 0.05°")
