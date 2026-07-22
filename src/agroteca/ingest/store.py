"""Stage 5 — persistence into Postgres/pgvector.

`register_vector` teaches psycopg how to send numpy vectors to a VECTOR column.
The `tsv` keyword column is built with the language-agnostic 'simple' config so
Phase 3's hybrid lexical search matches exact tokens consistently across ES+EN.
"""
import psycopg
from pgvector.psycopg import register_vector

from agroteca.config import settings


def connect() -> psycopg.Connection:
    conn = psycopg.connect(settings.db_url)
    register_vector(conn)
    return conn


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
