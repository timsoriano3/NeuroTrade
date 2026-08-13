"""Storage adapters: the local corpus.

Parquet on NVMe queried through DuckDB today; object storage when the corpus
outgrows the disk (§19). Callers see `StoragePort` and never learn which.
"""
