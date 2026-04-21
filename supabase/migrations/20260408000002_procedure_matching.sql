-- Procedure matching infrastructure.
-- Alias table, match_status columns, and cleanup of legacy columns.

-- 1. procedure_aliases table
-- Allows same alias for multiple procedures (e.g. "Fit for 55" → 13 procedures)
CREATE TABLE IF NOT EXISTS public.procedure_aliases (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  procedure_id text NOT NULL,
  alias text NOT NULL,
  alias_type text,  -- 'acronym', 'short_name', 'informal', 'package'
  created_at timestamptz DEFAULT now(),
  CONSTRAINT procedure_aliases_pkey PRIMARY KEY (id),
  CONSTRAINT procedure_aliases_procedure_fkey
    FOREIGN KEY (procedure_id) REFERENCES public.procedures(id) ON DELETE CASCADE,
  CONSTRAINT procedure_aliases_unique UNIQUE (alias, procedure_id)
);

CREATE INDEX IF NOT EXISTS idx_procedure_aliases_alias
  ON public.procedure_aliases (lower(alias));
CREATE INDEX IF NOT EXISTS idx_procedure_aliases_procedure
  ON public.procedure_aliases (procedure_id);

-- RLS
ALTER TABLE public.procedure_aliases ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Procedure aliases are viewable by everyone"
  ON public.procedure_aliases FOR SELECT USING (true);

CREATE POLICY "Service role can manage procedure aliases"
  ON public.procedure_aliases FOR ALL
  USING (auth.role() = 'service_role');

-- 2. match_status columns on meeting tables
-- Same pattern as dedup_status on organizations:
-- NULL = untried, 'matched' = has link(s), 'no_match' = tried but nothing found
ALTER TABLE public.lobbying_meetings
  ADD COLUMN IF NOT EXISTS match_status text;

ALTER TABLE public.commission_meetings
  ADD COLUMN IF NOT EXISTS match_status text;

-- Partial indexes for fast lookup of unprocessed meetings
CREATE INDEX IF NOT EXISTS idx_lobbying_meetings_match_status
  ON public.lobbying_meetings (match_status)
  WHERE match_status IS NULL;

CREATE INDEX IF NOT EXISTS idx_commission_meetings_match_status
  ON public.commission_meetings (match_status)
  WHERE match_status IS NULL;

-- 3. Drop legacy columns from commission_meetings
-- All match data now lives in meeting_procedure_links exclusively.
-- Drop index first, then columns.
DROP INDEX IF EXISTS idx_commission_meetings_procedure;

ALTER TABLE public.commission_meetings
  DROP COLUMN IF EXISTS matched_procedure_id,
  DROP COLUMN IF EXISTS match_confidence,
  DROP COLUMN IF EXISTS match_method;

-- 4. Trigram index on procedure titles for similarity search
CREATE INDEX IF NOT EXISTS idx_procedures_title_trgm
  ON public.procedures USING gin (title gin_trgm_ops);

-- 5. Similarity RPC for fallback server-side search
CREATE OR REPLACE FUNCTION match_procedure_by_similarity(
  query_text text,
  similarity_threshold float DEFAULT 0.3,
  max_results int DEFAULT 5
)
RETURNS TABLE (
  id text,
  title text,
  procedure_type text,
  proposal_date date,
  decision_date date,
  similarity_score float
)
LANGUAGE sql STABLE
AS $$
  SELECT
    p.id,
    p.title,
    p.procedure_type,
    p.proposal_date,
    p.decision_date,
    similarity(p.title, query_text) AS similarity_score
  FROM procedures p
  WHERE similarity(p.title, query_text) >= similarity_threshold
    AND p.is_deleted = false
  ORDER BY similarity_score DESC
  LIMIT max_results;
$$;
