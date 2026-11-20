"""Load the synthetic dataset into MongoDB.

A development tool, like its MySQL counterpart.  Nothing under
``piiscanner/connectors`` or ``piiscanner/classify`` may import it.

Usage::

    python -m piiscanner.testdata.load_mongodb --url mongodb://127.0.0.1:27017/
"""

from __future__ import annotations

import sys
import time

import click
from pymongo import MongoClient

from piiscanner.testdata.generate import (
    DATABASE_NAME,
    MONGO_COLLECTIONS,
    SEED,
    generate_documents,
)

BATCH_SIZE = 2000


def load(url: str, seed: int = SEED, quiet: bool = False) -> None:
    """Drop and rebuild the synthetic database."""
    client = MongoClient(url, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
        client.drop_database(DATABASE_NAME)
        database = client[DATABASE_NAME]
        for collection in MONGO_COLLECTIONS:
            started = time.monotonic()
            target = database[collection.name]
            batch = []
            written = 0
            for document in generate_documents(collection, seed):
                batch.append(document)
                if len(batch) >= BATCH_SIZE:
                    target.insert_many(batch, ordered=False)
                    written += len(batch)
                    batch.clear()
            if batch:
                target.insert_many(batch, ordered=False)
                written += len(batch)
            if not quiet:
                click.echo(
                    f"  {collection.name:<18} {written:>7,} docs  "
                    f"({time.monotonic() - started:.1f}s)  {collection.comment}"
                )
    finally:
        client.close()


@click.command()
@click.option(
    "--url",
    default="mongodb://127.0.0.1:27017/",
    show_default=True,
    help="Admin connection string. Never used by the scanner.",
)
@click.option("--seed", default=SEED, show_default=True)
def main(url: str, seed: int) -> None:
    """Rebuild the synthetic PII dataset in MongoDB."""
    click.echo(f"Building {DATABASE_NAME} (seed={seed})")
    started = time.monotonic()
    load(url, seed)
    click.echo(f"Done in {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
