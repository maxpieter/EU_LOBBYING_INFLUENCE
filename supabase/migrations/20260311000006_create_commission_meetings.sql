-- Commission meetings table (NEW - not in parl8)
-- Data source: EC Transparency Initiative (scraped) + Meeting Minutes PDFs
DROP TABLE IF EXISTS public.commission_meeting_organizations CASCADE;
DROP TABLE IF EXISTS public.commission_meetings CASCADE;

CREATE TABLE IF NOT EXISTS public.commission_meetings (
  id text NOT NULL,
  -- Host commissioner
  actor_id text,  -- FK to actors table
  commissioner_name text NOT NULL,
  commissioner_portfolio text,
  host_id text,  -- UUID from ec.europa.eu/transparencyinitiative URL
  -- Meeting details
  meeting_type text DEFAULT 'commissioner'::text,  -- 'commissioner' or 'cabinet'
  meeting_date date,
  location text,
  subject text,
  -- From meeting minutes PDF (nullable — not all meetings have minutes)
  commission_representatives jsonb DEFAULT '[]'::jsonb,
  organizations_raw text,  -- raw org names from HTML table
  transparency_register_ids text[] DEFAULT '{}'::text[],
  points_raised text,
  conclusions text,
  ares_number text,
  minutes_url text,
  -- Semantic linking (populated later by matching pipeline)
  matched_procedure_id text,
  match_confidence double precision,
  match_method text,
  -- Metadata
  source_url text,
  raw_data jsonb,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT commission_meetings_pkey PRIMARY KEY (id),
  CONSTRAINT commission_meetings_actor_fkey
    FOREIGN KEY (actor_id) REFERENCES public.actors(actor_id) ON DELETE SET NULL
);

-- Junction table linking commission meetings to organizations
CREATE TABLE IF NOT EXISTS public.commission_meeting_organizations (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  meeting_id text NOT NULL,
  organization_id text,
  organization_name text NOT NULL,
  eu_transparency_register_id text,
  CONSTRAINT commission_meeting_organizations_pkey PRIMARY KEY (id),
  CONSTRAINT commission_meeting_organizations_meeting_fkey
    FOREIGN KEY (meeting_id) REFERENCES public.commission_meetings(id) ON DELETE CASCADE,
  CONSTRAINT commission_meeting_organizations_org_fkey
    FOREIGN KEY (organization_id) REFERENCES public.organizations(id) ON DELETE SET NULL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_commission_meetings_actor ON public.commission_meetings (actor_id);
CREATE INDEX IF NOT EXISTS idx_commission_meetings_date ON public.commission_meetings (meeting_date);
CREATE INDEX IF NOT EXISTS idx_commission_meetings_commissioner ON public.commission_meetings (commissioner_name);
CREATE INDEX IF NOT EXISTS idx_commission_meetings_host ON public.commission_meetings (host_id);
CREATE INDEX IF NOT EXISTS idx_commission_meetings_procedure ON public.commission_meetings (matched_procedure_id);
CREATE INDEX IF NOT EXISTS idx_commission_meeting_orgs_meeting ON public.commission_meeting_organizations (meeting_id);
CREATE INDEX IF NOT EXISTS idx_commission_meeting_orgs_org ON public.commission_meeting_organizations (organization_id);
CREATE INDEX IF NOT EXISTS idx_commission_meeting_orgs_tr ON public.commission_meeting_organizations (eu_transparency_register_id);

CREATE TRIGGER commission_meetings_updated_at
  BEFORE UPDATE ON public.commission_meetings
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- RLS
ALTER TABLE public.commission_meetings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.commission_meeting_organizations ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Commission meetings are viewable by everyone"
  ON public.commission_meetings FOR SELECT
  USING (true);

CREATE POLICY "Service role can manage commission meetings"
  ON public.commission_meetings FOR ALL
  USING (auth.role() = 'service_role');

CREATE POLICY "Commission meeting organizations are viewable by everyone"
  ON public.commission_meeting_organizations FOR SELECT
  USING (true);

CREATE POLICY "Service role can manage commission meeting organizations"
  ON public.commission_meeting_organizations FOR ALL
  USING (auth.role() = 'service_role');
