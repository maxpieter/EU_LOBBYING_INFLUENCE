-- Actors table (matches parl8 schema)
CREATE TABLE IF NOT EXISTS public.actors (
  actor_id text NOT NULL,
  "fullName" text NOT NULL,
  actor_type text NOT NULL,
  profile_url text,
  image_url text,
  role text,
  country text,
  portfolio text,
  term_start date,
  term_end date,
  contacts jsonb,
  responsibilities jsonb,
  biography jsonb DEFAULT '[]'::jsonb,
  team jsonb DEFAULT '[]'::jsonb,
  declarations jsonb DEFAULT '[]'::jsonb,
  past_meetings jsonb DEFAULT '[]'::jsonb,
  speeches jsonb DEFAULT '[]'::jsonb,
  latest_news jsonb DEFAULT '[]'::jsonb,
  calendar jsonb DEFAULT '[]'::jsonb,
  documents jsonb DEFAULT '[]'::jsonb,
  transparency jsonb,
  -- Gold AI fields (kept null for parl8 compatibility)
  role_summary text,
  key_topics text[],
  declarations_summary text,
  -- embedding vector(1536),  -- Requires pgvector extension, add later if needed
  embedding_model text,
  -- Metadata
  parliament text DEFAULT 'eu'::text,
  description text,
  status text DEFAULT 'active'::text CHECK (status = ANY (ARRAY['active'::text, 'inactive'::text])),
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT actors_pkey PRIMARY KEY (actor_id)
);

CREATE TRIGGER actors_updated_at
  BEFORE UPDATE ON public.actors
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- RLS
ALTER TABLE public.actors ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Actors are viewable by everyone"
  ON public.actors FOR SELECT
  USING (true);

CREATE POLICY "Service role can manage actors"
  ON public.actors FOR ALL
  USING (auth.role() = 'service_role');
