import ollama
from agroteca.ingest import store
from agroteca.retrieve.rerank import rerank_search

SYSTEM_PROMPT = """You are an agronomy assistant. Answer the QUESTION using ONLY the CONTEXT chunks provided.

RULES:
1. GROUND — base every statement only on the CONTEXT. Never use outside knowledge.
2. CITE — after each claim, name its source in parentheses, e.g. (Libro_INIA_04.pdf).
3. ABSTAIN — if the CONTEXT does not contain the answer, reply with EXACTLY this line and nothing else:
   No encuentro la respuesta en el contexto disponible.
   Never invent an answer. Never bend this rule to please the user.
4. Answer in the same language as the QUESTION."""


def format_context(rows) -> str:
    """rows: list of (chunk_id, source_file, text) from rerank_search.
    Return ONE string: each chunk labeled with its source, blank line between chunks.
    """
    def format_chunk(chunk_id, source_file, text):
        # chunk_id is ignored for the final context (kept for traceability if needed later)
        return f"[Fuente: {source_file}]\n{text.strip()}"

    chunks = [format_chunk(*row) for row in rows]
    return "\n\n".join(chunks)


def ask_ollama(system: str, user: str) -> str:
    """Send a system+user prompt to the local model, return its text reply."""
    resp = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp["message"]["content"]


def answer(conn, question: str, k: int = 5) -> str:
    """Retrieve the top-k chunks for the question, then generate a grounded, cited answer."""
    rows = rerank_search(conn, question, k=k)              # 1. real chunks from the corpus
    context = format_context(rows)                         # 2. label each with its source
    user = f"CONTEXT:\n{context}\n\nQUESTION: {question}"  # 3. build the user message
    return ask_ollama(SYSTEM_PROMPT, user)                 # 4. ground + cite (or abstain)

def answer_stream(conn, question: str, k: int = 5):
    """Like answer(), but YIELDS the reply token-by-token as the model generates it."""
    rows = rerank_search(conn, question, k=k)              # retrieval (the pre-token wait)
    context = format_context(rows)
    user = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    stream = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        stream=True,                                       # <- Ollama now yields chunks, not one blob
    )
    for chunk in stream:
        yield chunk["message"]["content"]                  # <- hand over each token as it arrives

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # clean ES accents in the console

    conn = store.connect()
    question = "Según la guía de cosecha de agua lluvia, ¿cuánta agua se puede captar por cada milímetro de lluvia caída sobre un metro cuadrado de techo?"
    print(answer(conn, question))
    conn.close()