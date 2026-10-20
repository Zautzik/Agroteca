"""Query embedding for the hosted profile: the SAME MiniLM model as the index, run through ONNX
Runtime instead of torch, so the serverless bundle stays small.

Parity is a claim to measure, not assume: eval/parity_check.py compares these vectors against the
sentence-transformers reference on real queries and chunks, and the hosted profile is only
deployed once that check passes. The pipeline reproduced here is the model's own
sentence-transformers config: tokenize (truncate at max_seq_length=128, special tokens included)
-> transformer -> mean pooling over the attention mask. No normalization step, matching the
reference (pgvector cosine distance is scale-invariant anyway).
"""
from functools import lru_cache

import numpy as np

from agroteca.config import settings


def mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean of the token vectors that the attention mask marks as real (not padding).

    last_hidden_state: (batch, tokens, dim); attention_mask: (batch, tokens). Returns (batch, dim).
    """
    mask = attention_mask[..., None].astype(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    return summed / counts


@lru_cache(maxsize=1)
def _runtime():
    """Download (once per cold instance) and load the tokenizer + ONNX session."""
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    common = dict(repo_id=settings.embed_model, revision=settings.onnx_model_revision,
                  cache_dir=str(settings.onnx_cache_dir))
    model_path = hf_hub_download(filename=settings.onnx_model_file, **common)
    tokenizer = Tokenizer.from_file(hf_hub_download(filename="tokenizer.json", **common))
    tokenizer.enable_truncation(max_length=settings.embed_max_tokens)
    tokenizer.no_padding()

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1      # one query at a time; avoid oversubscribing a small vCPU
    session = ort.InferenceSession(model_path, sess_options=opts,
                                   providers=["CPUExecutionProvider"])
    input_names = {i.name for i in session.get_inputs()}
    outputs = [o.name for o in session.get_outputs()]
    # The token-level output, by name when the export labels it; else the first output.
    hidden_name = "last_hidden_state" if "last_hidden_state" in outputs else outputs[0]
    return tokenizer, session, input_names, hidden_name


def embed_texts_onnx(texts: list[str]) -> np.ndarray:
    """Embed texts one at a time (no padding, so no padding can leak into the mean)."""
    tokenizer, session, input_names, hidden_name = _runtime()
    out = []
    for text in texts:
        enc = tokenizer.encode(text)
        feeds = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
            "token_type_ids": np.array([enc.type_ids], dtype=np.int64),
        }
        hidden = session.run([hidden_name], {k: v for k, v in feeds.items() if k in input_names})[0]
        out.append(mean_pool(hidden, feeds["attention_mask"])[0])
    return np.stack(out).astype(np.float32)


def embed_query_onnx(text: str) -> np.ndarray:
    return embed_texts_onnx([text])[0]
