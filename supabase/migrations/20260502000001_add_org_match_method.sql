-- Streamline org matching audit trail to mirror meeting_procedure_links.
--
-- Adds match_method (parallels meeting_procedure_links.match_method) and
-- matched_tr_id (the TR entity the matcher selected, always written even
-- when the stub's meetings get relinked to a canonical org). This makes
-- AI accuracy directly evaluable via:
--   SELECT * FROM organizations WHERE match_method='ai_high' ORDER BY random() LIMIT 400;
--
-- Allowed match_method values (mirror procedure side):
--   NULL                  = not yet processed
--   'prefiltered'         = _should_skip pattern hit (govt / UN / ECB / etc.)
--   'fuzzy_auto_accept'   = fuzzy score >= auto_accept_threshold (deterministic)
--   'ai_high'             = AI returned "high" (mirrors procedure ai_high)
--   'no_match'            = no fuzzy candidates OR AI returned non-high
--
-- The legacy dedup_status column is intentionally left in place to avoid
-- breaking any existing readers; new pipeline code writes match_method
-- instead. dedup_status will be dropped in a follow-up once readers migrate.

ALTER TABLE public.organizations
  ADD COLUMN IF NOT EXISTS match_method text;

ALTER TABLE public.organizations
  ADD COLUMN IF NOT EXISTS matched_tr_id text;

CREATE INDEX IF NOT EXISTS idx_organizations_match_method
  ON public.organizations (match_method)
  WHERE match_method IS NULL;

COMMENT ON COLUMN public.organizations.match_method IS
  'Org→TR matcher cascade step: prefiltered | fuzzy_auto_accept | ai_high | no_match | NULL';

COMMENT ON COLUMN public.organizations.matched_tr_id IS
  'TR ID the matcher selected (always written for fuzzy_auto_accept and ai_high, even when meetings are relinked to a different canonical row)';
