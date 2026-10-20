"""Apply the SQL migrations in order to the database at AGROTECA_DB_URL.

For hosts where `psql` isn't at hand (e.g. a Windows machine pointing at a managed Postgres such
as Neon). Every migration is idempotent (IF NOT EXISTS / in-place UPDATE), so re-running is safe.

    uv run python scripts/apply_migrations.py                 # 001, 002, 003
    uv run python scripts/apply_migrations.py 003_public_demo # just one
"""
import sys
from pathlib import Path

import psycopg

from agroteca.config import settings

MIGRATIONS = Path(__file__).resolve().parents[1] / "migrations"


def main(names: list[str]) -> None:
    files = sorted(MIGRATIONS.glob("*.sql"))
    if names:
        files = [f for f in files if f.stem in names]
        if len(files) != len(names):
            sys.exit(f"unknown migration(s): {sorted(set(names) - {f.stem for f in files})}")
    with psycopg.connect(settings.db_url, autocommit=True) as conn:
        for f in files:
            conn.execute(f.read_text(encoding="utf-8"))
            print(f"applied {f.name}")


if __name__ == "__main__":
    main(sys.argv[1:])
