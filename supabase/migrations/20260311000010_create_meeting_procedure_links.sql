-- Meeting-to-procedure links junction table
-- Stores matches between meetings (lobbying + commission) and legislative procedures

CREATE TABLE IF NOT EXISTS public.meeting_procedure_links (
  id uuid NOT NULL DEFAULT gen_random_uuid(),

  -- Polymorphic: exactly one of these should be non-null
  lobbying_meeting_id text,
  commission_meeting_id text,

  -- The matched procedure
  procedure_id text NOT NULL,

  -- Match metadata
  match_method text NOT NULL,
  match_confidence float NOT NULL,
  match_rank integer DEFAULT 1,
  is_primary boolean DEFAULT false,
  match_details jsonb,

  created_at timestamptz DEFAULT now(),

  CONSTRAINT meeting_procedure_links_pkey PRIMARY KEY (id),
  CONSTRAINT meeting_procedure_links_procedure_fkey
    FOREIGN KEY (procedure_id) REFERENCES public.procedures(id),
  CONSTRAINT meeting_procedure_links_lobbying_fkey
    FOREIGN KEY (lobbying_meeting_id) REFERENCES public.lobbying_meetings(id),
  CONSTRAINT meeting_procedure_links_commission_fkey
    FOREIGN KEY (commission_meeting_id) REFERENCES public.commission_meetings(id),
  CONSTRAINT meeting_procedure_links_one_source_check
    CHECK (
      (lobbying_meeting_id IS NOT NULL AND commission_meeting_id IS NULL)
      OR (lobbying_meeting_id IS NULL AND commission_meeting_id IS NOT NULL)
    )
);

CREATE INDEX idx_mpl_lobbying ON meeting_procedure_links (lobbying_meeting_id) WHERE lobbying_meeting_id IS NOT NULL;
CREATE INDEX idx_mpl_commission ON meeting_procedure_links (commission_meeting_id) WHERE commission_meeting_id IS NOT NULL;
CREATE INDEX idx_mpl_procedure ON meeting_procedure_links (procedure_id);
CREATE INDEX idx_mpl_method ON meeting_procedure_links (match_method);
CREATE INDEX idx_mpl_confidence ON meeting_procedure_links (match_confidence);
