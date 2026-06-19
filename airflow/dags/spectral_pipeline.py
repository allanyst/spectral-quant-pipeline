from __future__ import annotations
import sys
sys.path.insert(0, "/opt/airflow/src")
sys.path.insert(0, "/opt/airflow/models")
sys.path.insert(0, "/opt/airflow")

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta


def run_ingestion(**context):
    from generate.synthetic import generate_samples
    from ingestion.loaders import get_connection, init_bronze, load_spectra
    records = generate_samples()
    conn = get_connection()
    init_bronze(conn)
    n = load_spectra(conn, records)
    conn.close()
    print(f"Loaded {n} new spectra into bronze ({len(records)} generated)")


def run_pls_training(**context):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pls_calibration", "/opt/airflow/models/pls_calibration.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    metrics = mod.train_and_save()
    for el, m in metrics.items():
        print(f"  {el.upper()}: R²={m['r2']:.4f}  RMSE={m['rmse']:.6f}")


def report_to_uqaas(**context):
    """
    POST full uncertainty budget to UQaaS for sodium using:
    - RMSECV from latest PLS training (models/rmsecv.json)
    - Representative Na measurements (3 repeats)
    - CRM certificate uncertainty
    """
    import json, urllib.request
    from pathlib import Path

    rmsecv_path = Path("/opt/airflow/models/rmsecv.json")
    if not rmsecv_path.exists():
        print("rmsecv.json not found — skipping UQaaS report")
        return

    with open(rmsecv_path) as f:
        rmsecv = json.load(f)

    uqaas_url = "http://uqaas-api:8000/wdxrf/full_budget"

    payload = {
        "element": "Na",
        "measurements": [0.142, 0.141, 0.143],   # normalized intensity (3 repeats)
        "rmsecv": rmsecv["na"],
        "u_cert": 0.005,                          # CRM u_cert (k=2 certificate)
        "k_cert": 2.0,
        "n_simulations": 50_000,
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        uqaas_url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read())
        print(f"UQaaS Na budget id={result['id']}")
        print(f"  u_combined    = {result['u_combined']:.6f}")
        print(f"  u_expanded(k=2) = {result['u_expanded_k2']:.6f}")
        for c in result["components"]:
            print(f"  {c['name']}: u={c['u_standard']:.6f} ({c['fraction_pct']:.1f}%)")
    except Exception as exc:
        # Non-blocking: UQaaS being down must not fail the pipeline
        print(f"UQaaS report skipped: {exc}")


with DAG(
    dag_id="spectral_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule_interval="0 7 * * 1-5",
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(seconds=10)},
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest_bronze",
        python_callable=run_ingestion,
    )

    dbt_run_task = BashOperator(
        task_id="dbt_run",
        bash_command="docker exec spectral-quant-pipeline-dbt-1 dbt run --profiles-dir /root/.dbt",
    )

    dbt_test_task = BashOperator(
        task_id="dbt_test",
        bash_command="docker exec spectral-quant-pipeline-dbt-1 dbt test --profiles-dir /root/.dbt",
    )

    pls_task = PythonOperator(
        task_id="train_pls",
        python_callable=run_pls_training,
    )

    uqaas_task = PythonOperator(
        task_id="report_to_uqaas",
        python_callable=report_to_uqaas,
        retries=1,
        retry_delay=timedelta(seconds=5),
    )

    ingest_task >> dbt_run_task >> dbt_test_task >> pls_task >> uqaas_task
