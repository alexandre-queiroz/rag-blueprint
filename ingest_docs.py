"""Ingestion script — indexes project documentation into Chroma Cloud.

Documents ingested:
  - docs/architecture.md       (system design overview)
  - docs/adrs/*.md             (10 Architecture Decision Records)

Usage:
    uv run python ingest_docs.py

Credentials are read from .env (copy .env.example → .env and fill in values).
"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rag.config import load_config
from rag.ingestion.pipeline import config_to_chunk_config, ingest_file
from rag.vector_db.client import get_client, get_collection

_DOCS_ROOT = Path(__file__).parent / "docs"

_FILES = [
    _DOCS_ROOT / "architecture.md",
    *sorted((_DOCS_ROOT / "adrs").glob("*.md")),
]


def main() -> None:
    config = load_config()
    chunk_config = config_to_chunk_config(config.ingestion)

    print(f"Connecting to Chroma Cloud (collection: {config.vector_db.collection}) …")
    client = get_client()
    collection = get_collection(client, config.vector_db.collection)
    print("Connected.\n")

    total_chunks = 0
    for path in _FILES:
        if not path.exists():
            print(f"  [SKIP]  {path.name} not found")
            continue
        count = ingest_file(collection, path, chunk_config)
        total_chunks += count
        print(f"  [OK]    {path.name}  →  {count} chunk(s)")

    print(f"\nDone. {len(_FILES)} files, {total_chunks} chunks indexed.")
    print(f"Collection: {config.vector_db.collection}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
