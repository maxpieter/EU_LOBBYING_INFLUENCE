-- Drop the match_confidence column from meeting_procedure_links.
--
-- The matcher hardcoded 0.9/0.95/1.0 by cascade step. Every ai_high row got
-- the same 0.9 regardless of the AI's actual confidence, so the float
-- carried no per-row information beyond what match_method already encodes.
-- Removing the column outright is cleaner than leaving a vestigial nullable
-- field. Historical rows lose their (uninformative) values.

DROP INDEX IF EXISTS public.idx_mpl_confidence;

ALTER TABLE public.meeting_procedure_links
  DROP COLUMN IF EXISTS match_confidence;
