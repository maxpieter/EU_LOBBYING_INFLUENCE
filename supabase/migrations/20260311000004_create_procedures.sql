-- Procedures table (matches parl8 schema)
CREATE TABLE IF NOT EXISTS public.procedures (
  id text NOT NULL,
  process_id text NOT NULL,
  title text NOT NULL,
  description text,
  procedure_type text NOT NULL,
  legal_basis text[] DEFAULT '{}'::text[],
  policy_area text,
  subjects text[] DEFAULT '{}'::text[],
  commission_document text,
  amending_acts jsonb DEFAULT '[]'::jsonb,
  background_documents jsonb DEFAULT '[]'::jsonb,
  celex_number text,
  status text,
  stage text,
  proposal_date date,
  last_activity_date date,
  decision_date date,
  events jsonb DEFAULT '[]'::jsonb,
  actors jsonb DEFAULT '[]'::jsonb,
  foreseen_activities jsonb DEFAULT '[]'::jsonb,
  -- Gold AI fields (kept null for parl8 compatibility)
  ai_summary text,
  ai_impact_analysis jsonb,
  ai_next_steps text,
  -- embedding vector(1536),  -- Requires pgvector extension, add later if needed
  embedding_model text,
  -- URLs
  api_uri text,
  oeil_url text,
  eurlex_proposal_url text,
  eurlex_final_act_url text,
  -- Soft delete
  is_deleted boolean DEFAULT false,
  deleted_at timestamp with time zone,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT procedures_pkey PRIMARY KEY (id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_procedures_process_id ON public.procedures (process_id);
CREATE INDEX IF NOT EXISTS idx_procedures_type ON public.procedures (procedure_type);
CREATE INDEX IF NOT EXISTS idx_procedures_status ON public.procedures (status);
CREATE INDEX IF NOT EXISTS idx_procedures_policy_area ON public.procedures (policy_area);
CREATE INDEX IF NOT EXISTS idx_procedures_proposal_date ON public.procedures (proposal_date);
CREATE INDEX IF NOT EXISTS idx_procedures_subjects ON public.procedures USING gin (subjects);

CREATE TRIGGER procedures_updated_at
  BEFORE UPDATE ON public.procedures
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- RLS
ALTER TABLE public.procedures ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Procedures are viewable by everyone"
  ON public.procedures FOR SELECT
  USING (true);

CREATE POLICY "Service role can manage procedures"
  ON public.procedures FOR ALL
  USING (auth.role() = 'service_role');
