-- Enable pg_trgm for fuzzy org name matching at ingestion time
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- GIN index for fast trigram similarity queries
CREATE INDEX IF NOT EXISTS idx_organizations_name_trgm
  ON public.organizations USING gin (name gin_trgm_ops);

-- RPC function: find canonical orgs similar to a given name
CREATE OR REPLACE FUNCTION match_org_by_similarity(
  query_name text,
  similarity_threshold float DEFAULT 0.3,
  max_results int DEFAULT 5
)
RETURNS TABLE (
  id text,
  name text,
  eu_transparency_register_id text,
  acronym text,
  country text,
  organization_type text,
  interests_represented text,
  similarity_score float
)
LANGUAGE sql STABLE
AS $$
  SELECT
    o.id,
    o.name,
    o.eu_transparency_register_id,
    o.acronym,
    o.country,
    o.organization_type,
    o.interests_represented,
    similarity(lower(o.name), lower(query_name))::float AS similarity_score
  FROM public.organizations o
  WHERE o.eu_transparency_register_id IS NOT NULL
    AND similarity(lower(o.name), lower(query_name)) >= similarity_threshold
  ORDER BY similarity_score DESC
  LIMIT max_results;
$$;
