-- Gold: one row per sample with calibration metadata and PLS input features
-- This is the table the PLS model reads for training and the API reads for lookup

{{
    config(
        materialized = 'table',
        schema = 'gold'
    )
}}

WITH peaks AS (
    SELECT * FROM {{ ref('fact_spectral_peaks') }}
),

-- Total intensity over the full spectrum (useful normalization feature for PLS)
total_intensity AS (
    SELECT
        spectrum_id,
        SUM(intensity) AS total_counts,
        AVG(intensity) AS mean_counts,
        STDDEV(intensity) AS std_counts
    FROM {{ ref('stg_xrf_spectra') }}
    GROUP BY spectrum_id
)

SELECT
    p.spectrum_id,
    p.sample_id,
    p.mix_type,
    -- True concentrations (training labels)
    COALESCE(p.conc_na, 0.0) AS conc_na,
    COALESCE(p.conc_mg, 0.0) AS conc_mg,
    COALESCE(p.conc_al, 0.0) AS conc_al,
    COALESCE(p.conc_fe, 0.0) AS conc_fe,
    COALESCE(p.conc_si, 0.0) AS conc_si,
    COALESCE(p.conc_p,  0.0) AS conc_p,
    -- Net peak features
    p.net_na, p.net_mg, p.net_al, p.net_fe, p.net_si, p.net_p,
    -- Spectral statistics
    ti.total_counts,
    ti.mean_counts,
    ti.std_counts,
    -- Derived: presence flags (element detected above background threshold)
    (p.net_na > 500)  AS has_na,
    (p.net_mg > 500)  AS has_mg,
    (p.net_al > 500)  AS has_al,
    (p.net_fe > 500)  AS has_fe,
    (p.net_si > 500)  AS has_si,
    (p.net_p  > 500)  AS has_p,
    p.acquired_at
FROM peaks p
JOIN total_intensity ti USING (spectrum_id)
ORDER BY p.sample_id
