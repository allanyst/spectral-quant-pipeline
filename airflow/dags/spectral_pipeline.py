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

    ingest_task >> dbt_run_task >> dbt_test_task >> pls_task
