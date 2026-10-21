Loaded and validated 22 golden questions (20 answerable+scorable).

## eval_pipeline — retrieval@5 against `https://agroteca.vercel.app` (concurrency=1, delay=20s)

| metric | value |
|---|---|
| questions scored | 20 |
| retrieval@5 accuracy | 70.0% (14/20) |
| p50 latency | 2143 ms |
| p95 latency | 3533 ms |
| mean latency | 2190 ms |
| max latency | 3780 ms |
| request errors | 0 |

**Misses:** q06, q07, q12, q15, q20, q22

q06, q07 and q12 have answers only in the copyrighted 'local' tier, which the public deployment
does not serve, so the live ceiling is 17/20. See the `hosted` row in eval/results.csv.
