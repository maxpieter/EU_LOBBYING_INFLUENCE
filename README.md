# EU Lobbying Influence

Code for MSc thesis on EU lobbying influence.

## Structure

```
analysis/    per-procedure analysis results (one folder per (COD) procedure)
pipeline/    Dagster pipeline that builds the integrated database
scripts/     reproducible scripts/notebooks, numbered in use-in-thesis order (thesis figures land in scripts/images/)
supabase/    schema.sql — the database structure as a single applyable file
```

Scripts under `scripts/` are numbered 00–07 to mirror the order the thesis introduces them (§4.1 through §4.4). See each script's docstring or the notebook headers for what they produce.

## Setup

```
uv sync
cp .env.example .env   # then fill in SUPABASE_* and ANTHROPIC_API_KEY
```

## Data

Raw bronze data (`data/`) is gitignored since it's too large and reproducible from the pipeline.
