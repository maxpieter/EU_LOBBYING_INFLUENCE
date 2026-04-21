-- Add dedup_status column to organizations table.
-- Tracks whether a stub org has been through fuzzy/AI classification,
-- so already-classified stubs are never re-sent to the AI.
--
-- Values:
--   NULL         = never tried (canonical TR org or new stub)
--   'prefiltered' = government body / institution, skip forever
--   'no_match'   = tried fuzzy + AI, no TR match found
--   'low'        = AI confidence too low
--   'medium'     = needs human review
--   'high'       = matched and applied (org should now have TR ID)

ALTER TABLE public.organizations
  ADD COLUMN IF NOT EXISTS dedup_status text;

-- Index for quick filtering of unprocessed stubs
CREATE INDEX IF NOT EXISTS idx_organizations_dedup_status
  ON public.organizations (dedup_status)
  WHERE dedup_status IS NULL AND eu_transparency_register_id IS NULL;

COMMENT ON COLUMN public.organizations.dedup_status IS
  'Fuzzy dedup classification: NULL=untried, prefiltered/no_match/low/medium/high';
