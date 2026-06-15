from fastapi import APIRouter, HTTPException
import psycopg2, psycopg2.extras, os

router = APIRouter(prefix="/samples", tags=["samples"])
DB_URL = os.environ.get("XRF_DB_URL", "postgresql://xrf:xrf_pass@postgres:5432/spectra_db")


def _conn():
    c = psycopg2.connect(DB_URL)
    c.autocommit = True
    return c


@router.get("/")
def list_samples(mix_type: str | None = None, limit: int = 50):
    sql = """
        SELECT sample_id, mix_type,
               conc_na, conc_mg, conc_al, conc_fe, conc_si, conc_p,
               net_na, net_mg, net_al, net_fe, net_si, net_p,
               total_counts, acquired_at::varchar
        FROM gold.fact_sample_summary
    """
    params: list = []
    if mix_type:
        sql += " WHERE mix_type = %s"
        params.append(mix_type)
    sql += f" ORDER BY sample_id LIMIT {limit}"

    conn = _conn()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.get("/{sample_id}")
def get_sample(sample_id: str):
    conn = _conn()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT gs.*, s.two_theta, s.intensities
            FROM gold.fact_sample_summary gs
            JOIN bronze.xrf_spectra s USING (sample_id)
            WHERE gs.sample_id = %s
        """, [sample_id])
        row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail=f"{sample_id} not found")
    return dict(row)
