-- Materialized views for common queries

-- Organization meeting stats
CREATE MATERIALIZED VIEW IF NOT EXISTS public.mv_organization_meeting_stats AS
SELECT
  o.id AS organization_id,
  o.name AS organization_name,
  o.organization_type,
  o.country,
  o.eu_transparency_register_id,
  COUNT(DISTINCT lm.id) AS total_meetings,
  COUNT(DISTINCT lm.mep_id) AS unique_meps_met,
  MIN(lm.meeting_date) AS first_meeting_date,
  MAX(lm.meeting_date) AS last_meeting_date,
  COUNT(DISTINCT lm.committee_acronym) FILTER (WHERE lm.committee_acronym IS NOT NULL) AS committees_reached
FROM public.organizations o
LEFT JOIN public.lobbying_meetings lm ON o.id = lm.organization_id
GROUP BY o.id, o.name, o.organization_type, o.country, o.eu_transparency_register_id;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_org_stats_id ON public.mv_organization_meeting_stats (organization_id);

-- MEP meeting stats
CREATE MATERIALIZED VIEW IF NOT EXISTS public.mv_mep_meeting_stats AS
SELECT
  m.id AS mep_id,
  m."fullName" AS mep_name,
  m."politicalGroup",
  m.country,
  COUNT(DISTINCT lm.id) AS total_meetings,
  COUNT(DISTINCT lm.organization_id) AS unique_organizations_met,
  MIN(lm.meeting_date) AS first_meeting_date,
  MAX(lm.meeting_date) AS last_meeting_date
FROM public.meps m
LEFT JOIN public.lobbying_meetings lm ON m.id = lm.mep_id
GROUP BY m.id, m."fullName", m."politicalGroup", m.country;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_mep_stats_id ON public.mv_mep_meeting_stats (mep_id);

-- Refresh function
CREATE OR REPLACE FUNCTION refresh_all_materialized_views()
RETURNS void AS $$
BEGIN
  REFRESH MATERIALIZED VIEW CONCURRENTLY public.mv_organization_meeting_stats;
  REFRESH MATERIALIZED VIEW CONCURRENTLY public.mv_mep_meeting_stats;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
