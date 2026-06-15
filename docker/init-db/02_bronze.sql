-- Bronze layer: raw XRF spectra stored as FLOAT[] arrays
-- two_theta: 2θ angles grid (shared by all spectra, stored once)
-- intensities: measured counts per channel, parallel to two_theta

CREATE TABLE IF NOT EXISTS bronze.xrf_spectra (
    spectrum_id     SERIAL PRIMARY KEY,
    sample_id       VARCHAR(32)     NOT NULL,
    mix_type        VARCHAR(16)     NOT NULL,   -- 'unary' | 'binary' | 'ternary'
    -- true concentrations (mass fraction, sums to 1)
    conc_na         FLOAT,
    conc_mg         FLOAT,
    conc_al         FLOAT,
    conc_fe         FLOAT,
    conc_si         FLOAT,
    conc_p          FLOAT,
    -- spectral data
    two_theta       FLOAT[]         NOT NULL,   -- 2θ grid, degrees
    intensities     FLOAT[]         NOT NULL,   -- counts per channel
    n_channels      INT             NOT NULL,
    -- metadata
    instrument_id   VARCHAR(16)     NOT NULL DEFAULT 'WDXRF-01',
    acquired_at     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    loaded_at       TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_array_length CHECK (cardinality(two_theta) = n_channels
                                   AND cardinality(intensities) = n_channels)
);

CREATE INDEX IF NOT EXISTS idx_xrf_sample ON bronze.xrf_spectra (sample_id);
CREATE INDEX IF NOT EXISTS idx_xrf_mix_type ON bronze.xrf_spectra (mix_type);
CREATE INDEX IF NOT EXISTS idx_xrf_acquired ON bronze.xrf_spectra (acquired_at);
