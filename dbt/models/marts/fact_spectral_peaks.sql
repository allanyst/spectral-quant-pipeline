-- Gold: peak intensities per element per spectrum
-- Aggregates Silver long-format rows within ±0.5° windows around each Kα peak
-- Result: one row per spectrum with net peak intensity for each element

{{
    config(
        materialized = 'table',
        schema = 'gold'
    )
}}

WITH base AS (
    SELECT DISTINCT
        spectrum_id,
        sample_id,
        mix_type,
        conc_na, conc_mg, conc_al, conc_fe, conc_si, conc_p,
        acquired_at
    FROM {{ ref('stg_xrf_spectra') }}
),

peak_windows AS (
    SELECT
        spectrum_id,
        -- Peak intensity = MAX in ±0.5° window around each Kα line
        -- Background estimated from flanking regions (±1.5° to ±2.5°)
        MAX(CASE WHEN ABS(two_theta - 55.1) <= 0.5 THEN intensity END) AS peak_na,
        AVG(CASE WHEN two_theta BETWEEN 52.5 AND 53.5 THEN intensity END) AS bg_na,

        MAX(CASE WHEN ABS(two_theta - 45.2) <= 0.5 THEN intensity END) AS peak_mg,
        AVG(CASE WHEN two_theta BETWEEN 42.5 AND 43.5 THEN intensity END) AS bg_mg,

        MAX(CASE WHEN ABS(two_theta - 57.5) <= 0.5 THEN intensity END) AS peak_al,
        AVG(CASE WHEN two_theta BETWEEN 55.0 AND 56.0 THEN intensity END) AS bg_al,

        MAX(CASE WHEN ABS(two_theta - 17.5) <= 0.5 THEN intensity END) AS peak_fe,
        AVG(CASE WHEN two_theta BETWEEN 14.5 AND 15.5 THEN intensity END) AS bg_fe,

        MAX(CASE WHEN ABS(two_theta - 75.0) <= 0.5 THEN intensity END) AS peak_si,
        AVG(CASE WHEN two_theta BETWEEN 72.5 AND 73.5 THEN intensity END) AS bg_si,

        MAX(CASE WHEN ABS(two_theta - 89.3) <= 0.5 THEN intensity END) AS peak_p,
        AVG(CASE WHEN two_theta BETWEEN 86.5 AND 87.5 THEN intensity END) AS bg_p
    FROM {{ ref('stg_xrf_spectra') }}
    GROUP BY spectrum_id
)

SELECT
    b.spectrum_id,
    b.sample_id,
    b.mix_type,
    b.conc_na,
    b.conc_mg,
    b.conc_al,
    b.conc_fe,
    b.conc_si,
    b.conc_p,
    b.acquired_at,
    -- Net peak intensity (peak minus local background)
    GREATEST(pw.peak_na - COALESCE(pw.bg_na, 800), 0) AS net_na,
    GREATEST(pw.peak_mg - COALESCE(pw.bg_mg, 800), 0) AS net_mg,
    GREATEST(pw.peak_al - COALESCE(pw.bg_al, 800), 0) AS net_al,
    GREATEST(pw.peak_fe - COALESCE(pw.bg_fe, 800), 0) AS net_fe,
    GREATEST(pw.peak_si - COALESCE(pw.bg_si, 800), 0) AS net_si,
    GREATEST(pw.peak_p  - COALESCE(pw.bg_p,  800), 0) AS net_p,
    -- Raw peaks (for diagnostics)
    pw.peak_na, pw.peak_mg, pw.peak_al, pw.peak_fe, pw.peak_si, pw.peak_p
FROM base b
JOIN peak_windows pw USING (spectrum_id)
