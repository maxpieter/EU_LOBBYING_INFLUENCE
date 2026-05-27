# EU Lobbying Influence

Code and data for an MSc thesis on EU lobbying influence on legislation.

## Structure

```
analysis/    per-procedure analysis results (one folder per (COD) procedure)
data/        raw bronze data fetched by the pipeline
images/      thesis figures (PDF)
pipeline/    Dagster pipeline that builds the integrated database
scripts/     reproducible scripts/notebooks, numbered in thesis order
supabase/    SQL migrations
```

Scripts under `scripts/` are numbered 00–07 to mirror the order the thesis introduces them (§4.1 through §4.4). See each script's docstring or the notebook headers for what they produce.

## Setup

```
uv sync
cp .env.example .env   # then fill in SUPABASE_* and ANTHROPIC_API_KEY
```

## Data

Raw bronze data (`data/`) and the Python virtualenv (`.venv/`) are gitignored — they are large and reproducible from the pipeline. Same for `.env` and any local Dagster temp directories.
