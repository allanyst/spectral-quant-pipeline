"""
Bronze layer loader: inserts synthetic XRF spectra into PostgreSQL FLOAT[] columns.
Uses psycopg2 with execute_values for batch inserts — handles array literals natively.
"""
from __future__ import annotations
import os
import psycopg2
from psycopg2.extras import execute_values

DB_URL = os.environ.get(
    "XRF_DB_URL",
    "postgresql://xrf:xrf_pass@localhost:5434/spectra_db"
)


def get_connection() -> psycopg2.extensions.connection:
    return psycopg2.connect(DB_URL)


def init_bronze(conn) -> None:
    """Ensure bronze table exists (idempotent — schema already created by init SQL)."""
    # Table is created by docker/init-db/02_bronze.sql; nothing to do here.
    conn.commit()


def load_spectra(conn, records: list[dict]) -> int:
    """
    Bulk-insert spectra records into bronze.xrf_spectra.
    Skips duplicates by sample_id (ON CONFLICT DO NOTHING).
    Returns number of rows actually inserted.
    """
    # Add UNIQUE on sample_id via ALTER if not already present
    with conn.cursor() as cur:
        cur.execute("""
            DO $$ BEGIN
                ALTER TABLE bronze.xrf_spectra ADD CONSTRAINT uq_xrf_sample_id UNIQUE (sample_id);
            EXCEPTION WHEN duplicate_table THEN NULL;
            END $$
        """)
    conn.commit()

    rows = [
        (
            r["sample_id"],
            r["mix_type"],
            r["conc_na"],
            r["conc_mg"],
            r["conc_al"],
            r["conc_fe"],
            r["conc_si"],
            r["conc_p"],
            r["two_theta"],
            r["intensities"],
            r["n_channels"],
        )
        for r in records
    ]

    sql = """
        INSERT INTO bronze.xrf_spectra
            (sample_id, mix_type,
             conc_na, conc_mg, conc_al, conc_fe, conc_si, conc_p,
             two_theta, intensities, n_channels)
        VALUES %s
        ON CONFLICT (sample_id) DO NOTHING
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, template=None, page_size=len(rows))
        inserted = cur.rowcount
    conn.commit()
    return inserted
