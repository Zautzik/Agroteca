# Hosted-profile parity check

Embedder `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` @ `main`, 44 texts (golden questions + reference answers).

## 1. Embedding parity (ONNX variant vs sentence-transformers)

| ONNX file | Size (MB) | Min cosine | Mean cosine | Pass |
|---|---|---|---|---|
| `onnx/model.onnx` | 470 | 1.00000 | 1.00000 | yes |
| `onnx/model_O1.onnx` | 470 | 1.00000 | 1.00000 | yes |
| `onnx/model_O2.onnx` | 470 | 1.00000 | 1.00000 | yes |
| `onnx/model_O3.onnx` | 470 | 1.00000 | 1.00000 | yes |
| `onnx/model_O4.onnx` | 235 | 1.00000 | 1.00000 | yes |
| `onnx/model_qint8_arm64.onnx` | 118 | 0.98811 | 0.99492 | no |
| `onnx/model_qint8_avx512.onnx` | 118 | 0.98811 | 0.99492 | no |
| `onnx/model_qint8_avx512_vnni.onnx` | 118 | 0.98811 | 0.99492 | no |
| `onnx/model_quint8_avx2.onnx` | 118 | 0.98758 | 0.99467 | no |

## 2. Dense retrieval parity (top-20 overlap with the reference)

| ONNX file | Mean overlap | Min overlap |
|---|---|---|
| `onnx/model.onnx` | 1.000 | 1.000 |
| `onnx/model_O1.onnx` | 1.000 | 1.000 |
| `onnx/model_O2.onnx` | 1.000 | 1.000 |
| `onnx/model_O3.onnx` | 1.000 | 1.000 |
| `onnx/model_O4.onnx` | 1.000 | 1.000 |

## 3. Rerank parity (local cross-encoder vs Jina API, same pools)

| Question | Spearman | Top-5 agreement | Local range | API range |
|---|---|---|---|---|
| q1 | 0.998 | 5/5 | 0.045–0.740 | 0.045–0.740 |
| q2 | 1.000 | 5/5 | 0.037–0.821 | 0.037–0.821 |
| q3 | 1.000 | 5/5 | 0.041–0.708 | 0.041–0.707 |
| q4 | 0.998 | 5/5 | 0.051–0.905 | 0.051–0.905 |
| q5 | 1.000 | 5/5 | 0.025–0.690 | 0.025–0.690 |

Passing ONNX variants (min cosine >= 0.999): `onnx/model.onnx`, `onnx/model_O1.onnx`, `onnx/model_O2.onnx`, `onnx/model_O3.onnx`, `onnx/model_O4.onnx`
