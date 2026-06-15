-- Silver: unnest FLOAT[] arrays → one row per (spectrum, channel)
-- This is the key transformation: wide array format → long analytical format
-- UNNEST with ordinality gives us channel_idx (1-based) for joining back to 2θ grid

{{
    config(
        materialized = 'view',
        schema = 'silver'
    )
}}

WITH unnested AS (
    SELECT
        spectrum_id,
        sample_id,
        mix_type,
        conc_na,
        conc_mg,
        conc_al,
        conc_fe,
        conc_si,
        conc_p,
        n_channels,
        instrument_id,
        acquired_at,
        channel_idx,
        two_theta_val,
        intensity_val
    FROM bronze.xrf_spectra,
         LATERAL UNNEST(two_theta, intensities)
             WITH ORDINALITY AS t(two_theta_val, intensity_val, channel_idx)
)

SELECT
    spectrum_id,
    sample_id,
    mix_type,
    -- concentrations (NULL means element absent in this sample)
    NULLIF(conc_na, 0.0)  AS conc_na,
    NULLIF(conc_mg, 0.0)  AS conc_mg,
    NULLIF(conc_al, 0.0)  AS conc_al,
    NULLIF(conc_fe, 0.0)  AS conc_fe,
    NULLIF(conc_si, 0.0)  AS conc_si,
    NULLIF(conc_p,  0.0)  AS conc_p,
    n_channels,
    instrument_id,
    acquired_at,
    channel_idx::INT       AS channel_idx,
    ROUND(two_theta_val::NUMERIC, 2) AS two_theta,
    ROUND(intensity_val::NUMERIC, 2) AS intensity
FROM unnested
