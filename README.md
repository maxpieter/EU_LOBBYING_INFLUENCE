# EU Lobbying Influence

Code for MSc thesis on EU lobbying influence.

## Structure

```
schema.sql           database schema as a single applyable file (psql -f schema.sql)
analysis/            alignment-pipeline scripts (produce the per-procedure dyads)
analysis_results/    per-procedure outputs (one folder per (COD) procedure)
pipeline/            Dagster pipeline that builds the integrated database
scripts/             numbered analysis scripts/notebooks; thesis figures land in scripts/images/
```

Scripts under `scripts/` are numbered 00–07 to mirror the order the thesis introduces them (§4.1 through §4.4). See each script's docstring or the notebook headers for what they produce.

## Setup

```
uv sync
cp .env.example .env   # then fill in SUPABASE_* and ANTHROPIC_API_KEY
```

## Data

Raw bronze data (`data/`) is gitignored since it's too large and reproducible from the pipeline.
