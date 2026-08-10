"""Stage 5 — persistence into Postgres/pgvector.

`register_vector` teaches psycopg how to send numpy vectors to a VECTOR column.
The `tsv` keyword column is built with the language-agnostic 'simple' config so
Phase 3's hybrid lexical search matches exact tokens consistently across ES+EN.
"""
import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from agroteca.config import settings

# TCP keepalives so idle connections survive Docker Desktop's WSL2 port-forward, which
# resets idle localhost connections (the 10053 abort). Keepalives help but the proxy can
# still drop; make_pool()'s check= is the real safety net (validates a conn on checkout).
KEEPALIVE = {"keepalives": 1, "keepalives_idle": 10, "keepalives_interval": 5, "keepalives_count": 5}


def connect() -> psycopg.Connection:
    """One-shot connection for CLI / ingest / eval scripts. The served API uses make_pool()."""
    conn = psycopg.connect(settings.db_url, **KEEPALIVE)
    register_vector(conn)
    return conn


def make_pool(min_size: int = 1, max_size: int | None = None) -> ConnectionPool:
    """A connection pool for the served API: reuses connections instead of a fresh TCP
    handshake + auth per request. `configure` runs register_vector on each pooled
    connection; `check` validates a connection on checkout and transparently replaces one
    the Docker proxy dropped while idle — so a request never receives a dead connection."""
    return ConnectionPool(
        settings.db_url,
        min_size=min_size,
        max_size=max_size or settings.db_pool_max,
        kwargs=KEEPALIVE,
        configure=register_vector,
        check=ConnectionPool.check_connection,   # SELECT 1 on checkout; discard + replace dead conns
        open=False,
    )


def upsert_document(conn: psycopg.Connection, doc: dict) -> None:
    conn.execute(
        """
        INSERT INTO documents (doc_id, source_file, title, tier, lang, topic, url)
        VALUES (%(doc_id)s, %(source_file)s, %(title)s, %(tier)s, %(lang)s, %(topic)s, %(url)s)
        ON CONFLICT (doc_id) DO UPDATE SET
            source_file = EXCLUDED.source_file, title = EXCLUDED.title, tier = EXCLUDED.tier,
            lang = EXCLUDED.lang, topic = EXCLUDED.topic, url = EXCLUDED.url
        """,
        doc,
    )


def wipe_chunks(conn: psycopg.Connection, doc_id: str) -> None:
    """Delete a document's chunks so re-ingesting is idempotent."""
    conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))


def insert_chunks(conn: psycopg.Connection, rows: list[dict]) -> None:
    """Bulk-insert chunk rows. The tsv keyword index uses the language-agnostic
    'simple' config so hybrid lexical search matches exact tokens across ES+EN."""
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks
                (chunk_id, doc_id, chunk_index, text, embedding, tsv,
                 tier, lang, topic, page, char_start, char_end)
            VALUES
                (%(chunk_id)s, %(doc_id)s, %(chunk_index)s, %(text)s, %(embedding)s,
                 to_tsvector('simple', %(text)s),
                 %(tier)s, %(lang)s, %(topic)s, %(page)s, %(char_start)s, %(char_end)s)
            """,
            rows,
        )
