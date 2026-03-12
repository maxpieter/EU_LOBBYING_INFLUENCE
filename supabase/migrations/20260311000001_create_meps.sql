-- MEPs table (matches parl8 schema exactly)
CREATE TABLE IF NOT EXISTS public.meps (
  id bigint NOT NULL,
  "fullName" text,
  country text,
  "politicalGroup" text,
  "nationalPoliticalGroup" text,
  profile_url text,
  image_url text,
  role text,
  birth_date date,
  birth_place text,
  socials jsonb DEFAULT '{}'::jsonb,
  committees jsonb DEFAULT '[]'::jsonb,
  navigation_links jsonb DEFAULT '{}'::jsonb,
  contacts jsonb DEFAULT '[]'::jsonb,
  cv jsonb DEFAULT '[]'::jsonb,
  assistants jsonb DEFAULT '[]'::jsonb,
  declarations jsonb DEFAULT '[]'::jsonb,
  past_meetings jsonb DEFAULT '[]'::jsonb,
  -- Gold AI fields (kept null for parl8 compatibility)
  declarations_summary text,
  speech_summary text DEFAULT ''::text,
  speech_top_words jsonb DEFAULT '[]'::jsonb,
  speech_sources jsonb DEFAULT '[]'::jsonb,
  -- Status and timestamps
  status text DEFAULT 'active'::text CHECK (status = ANY (ARRAY['active'::text, 'inactive'::text])),
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT meps_pkey PRIMARY KEY (id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_meps_country ON public.meps (country);
CREATE INDEX IF NOT EXISTS idx_meps_political_group ON public.meps ("politicalGroup");
CREATE INDEX IF NOT EXISTS idx_meps_status ON public.meps (status);

-- Updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER meps_updated_at
  BEFORE UPDATE ON public.meps
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- RLS
ALTER TABLE public.meps ENABLE ROW LEVEL SECURITY;

CREATE POLICY "MEPs are viewable by everyone"
  ON public.meps FOR SELECT
  USING (true);

CREATE POLICY "Service role can manage MEPs"
  ON public.meps FOR ALL
  USING (auth.role() = 'service_role');
