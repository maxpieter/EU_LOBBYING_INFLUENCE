-- Function to delete orphaned stub organisations.
-- Only deletes stubs (no TR ID) that no meeting references.
-- Canonical orgs (with TR ID) are kept even if unreferenced.
CREATE OR REPLACE FUNCTION cleanup_orphaned_stubs()
RETURNS integer
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
  deleted_count integer;
BEGIN
  DELETE FROM organizations o
  WHERE o.eu_transparency_register_id IS NULL
    AND NOT EXISTS (
      SELECT 1 FROM lobbying_meetings lm WHERE lm.organization_id = o.id
    )
    AND NOT EXISTS (
      SELECT 1 FROM commission_meeting_organizations cmo WHERE cmo.organization_id = o.id
    );
  GET DIAGNOSTICS deleted_count = ROW_COUNT;
  RETURN deleted_count;
END;
$$;
