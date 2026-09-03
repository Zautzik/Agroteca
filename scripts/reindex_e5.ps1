# reindex_e5.ps1 — resume the e5-large (1024-dim) re-index ONE small batch at a time.
# Run it whenever you have a few minutes; the PC can sleep between runs, and Ctrl-C is safe
# (each finished doc is already committed). Repeat until it reports 71/71.
#   .\reindex_e5.ps1            # next 3 docs
#   .\reindex_e5.ps1 -Batch 5  # next 5 docs (~30 min on CPU)
param([int]$Batch = 3)

$env:AGROTECA_DB_URL      = "postgresql://postgres:agroteca@localhost:5433/agroteca_e5"
$env:AGROTECA_EMBED_MODEL = "intfloat/multilingual-e5-large"
$env:AGROTECA_EMBED_DIM   = "1024"
$env:AGROTECA_LOCAL_MODE  = "1"

$before = (docker exec agroteca-pgvector psql -U postgres -d agroteca_e5 -tAc "SELECT count(DISTINCT doc_id) FROM chunks;").Trim()
Write-Host "e5 index: $before/71 docs done. Embedding the next $Batch (Ctrl-C is safe)..." -ForegroundColor Cyan
uv run python -m agroteca.ingest.run --resume --limit $Batch
$after = (docker exec agroteca-pgvector psql -U postgres -d agroteca_e5 -tAc "SELECT count(DISTINCT doc_id) FROM chunks;").Trim()
if ($after -eq "71") { Write-Host "e5 index: 71/71 -- DONE. Ready to eval." -ForegroundColor Green }
else { Write-Host "e5 index: $after/71 docs done. Run me again for the next batch." -ForegroundColor Green }
