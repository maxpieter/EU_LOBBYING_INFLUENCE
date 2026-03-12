-- Lobbying meetings table (matches parl8 schema)
CREATE TABLE IF NOT EXISTS public.lobbying_meetings (
  id text NOT NULL,
  mep_id integer,
  organization_id text,
  meeting_date date,
  title text,
  location text,
  capacity text,
  related_procedure text,
  committee_acronym text,
  meeting_purpose text,
  policy_area text,
  meeting_type character varying,
  transparency_level character varying,
  source_data jsonb,
  processed_at timestamp with time zone DEFAULT now(),
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT lobbying_meetings_pkey PRIMARY KEY (id),
  CONSTRAINT lobbying_meetings_mep_id_fkey
    FOREIGN KEY (mep_id) REFERENCES public.meps(id),
  CONSTRAINT lobbying_meetings_organization_id_fkey
    FOREIGN KEY (organization_id) REFERENCES public.organizations(id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_lobbying_meetings_mep_id ON public.lobbying_meetings (mep_id);
CREATE INDEX IF NOT EXISTS idx_lobbying_meetings_org_id ON public.lobbying_meetings (organization_id);
CREATE INDEX IF NOT EXISTS idx_lobbying_meetings_date ON public.lobbying_meetings (meeting_date);
CREATE INDEX IF NOT EXISTS idx_lobbying_meetings_committee ON public.lobbying_meetings (committee_acronym);

-- RLS
ALTER TABLE public.lobbying_meetings ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Lobbying meetings are viewable by everyone"
  ON public.lobbying_meetings FOR SELECT
  USING (true);

CREATE POLICY "Service role can manage lobbying meetings"
  ON public.lobbying_meetings FOR ALL
  USING (auth.role() = 'service_role');
