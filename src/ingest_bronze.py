"""One-shot script to generate synthetic spectra and load into Bronze."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.generate.synthetic import generate_samples
from src.ingestion.loaders import get_connection, init_bronze, load_spectra

records = generate_samples()
print(f"Generated {len(records)} spectra")

conn = get_connection()
init_bronze(conn)
n = load_spectra(conn, records)
conn.close()
print(f"Inserted {n} new rows into bronze.xrf_spectra")
