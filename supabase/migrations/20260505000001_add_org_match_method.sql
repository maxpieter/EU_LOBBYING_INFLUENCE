-- Persist the OrgResolver cascade outcome on each meeting-side org reference.
-- Mirrors meeting_procedure_links.match_method (procedure side), but for the
-- org→TR resolution. Vocabulary comes from OrgResolver.resolve():
--   tr_id_exact, name_exact, cleaned_name, cleaned_acronym, acronym,
--   parenthetical, parenthetical_acronym, prefix, tr_id_extracted,
--   stub, stub_empty.
-- For meetings whose stub was later promoted by eu_organizations_fuzzy, this
-- column stays at its silver value — the post-hoc classification lives on
-- organizations.match_method.

ALTER TABLE public.lobbying_meetings
  ADD COLUMN IF NOT EXISTS org_match_method text;

ALTER TABLE public.commission_meeting_organizations
  ADD COLUMN IF NOT EXISTS org_match_method text;

CREATE INDEX IF NOT EXISTS idx_lobbying_meetings_org_match_method
  ON public.lobbying_meetings (org_match_method);

CREATE INDEX IF NOT EXISTS idx_commission_meeting_orgs_org_match_method
  ON public.commission_meeting_organizations (org_match_method);
