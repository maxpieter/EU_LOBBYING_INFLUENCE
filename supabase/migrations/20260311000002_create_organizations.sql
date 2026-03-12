-- Organizations table (matches parl8 schema)
CREATE TABLE IF NOT EXISTS public.organizations (
  id text NOT NULL,
  name text NOT NULL,
  normalized_name text,
  official_name text,
  website text,
  organization_type text,
  industry_sector character varying,
  country character varying,
  eu_transparency_register_id text,
  description text,
  founding_year integer,
  employee_count_range character varying,
  annual_revenue_range character varying,
  total_meetings_count integer DEFAULT 0,
  unique_meps_met integer DEFAULT 0,
  influence_score numeric,
  transparency_score numeric,
  activity_level character varying,
  scraped_at timestamp with time zone,
  logo_url text,
  social_media jsonb DEFAULT '{}'::jsonb,
  key_personnel jsonb DEFAULT '[]'::jsonb,
  policy_focus_areas text[] DEFAULT '{}'::text[],
  acronym text,
  city text,
  address text,
  post_code text,
  level_of_interest text,
  interests_represented text,
  form_of_entity text,
  source_of_funding text,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT organizations_pkey PRIMARY KEY (id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_organizations_name ON public.organizations (name);
CREATE INDEX IF NOT EXISTS idx_organizations_transparency_id
  ON public.organizations (eu_transparency_register_id);
CREATE INDEX IF NOT EXISTS idx_organizations_type ON public.organizations (organization_type);
CREATE INDEX IF NOT EXISTS idx_organizations_country ON public.organizations (country);

CREATE TRIGGER organizations_updated_at
  BEFORE UPDATE ON public.organizations
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- RLS
ALTER TABLE public.organizations ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Organizations are viewable by everyone"
  ON public.organizations FOR SELECT
  USING (true);

CREATE POLICY "Service role can manage organizations"
  ON public.organizations FOR ALL
  USING (auth.role() = 'service_role');
