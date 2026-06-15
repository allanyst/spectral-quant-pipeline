FROM apache/airflow:2.8.1-python3.11
USER root
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*
USER airflow
RUN pip install --no-cache-dir \
    psycopg2-binary==2.9.9 \
    scikit-learn==1.5.0 \
    numpy==1.26.4
