-- Procedure articles and amendments (matches parl8 schema)
CREATE TABLE IF NOT EXISTS public.procedure_articles (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  procedure_id text NOT NULL,
  element_type text NOT NULL CHECK (element_type = ANY (ARRAY['recital'::text, 'article'::text])),
  element_number text NOT NULL,
  title text,
  content text NOT NULL,
  document_source text NOT NULL,
  document_version text NOT NULL CHECK (document_version = ANY (ARRAY['proposal'::text, 'committee'::text, 'adopted'::text])),
  sort_order integer DEFAULT 0,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT procedure_articles_pkey PRIMARY KEY (id),
  CONSTRAINT procedure_articles_procedure_id_fkey
    FOREIGN KEY (procedure_id) REFERENCES public.procedures(id)
);

CREATE INDEX IF NOT EXISTS idx_procedure_articles_procedure ON public.procedure_articles (procedure_id);
CREATE INDEX IF NOT EXISTS idx_procedure_articles_type ON public.procedure_articles (element_type);

CREATE TABLE IF NOT EXISTS public.procedure_amendments (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  procedure_id text NOT NULL,
  event_date date,
  document_id text NOT NULL,
  work_type text,
  amendment_number integer NOT NULL,
  target_element text,
  target_type text,
  original_text text,
  amended_text text,
  justification text,
  committee text,
  rapporteur_mep_id integer,
  submitted_by jsonb,
  adopted boolean,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT procedure_amendments_pkey PRIMARY KEY (id),
  CONSTRAINT procedure_amendments_procedure_id_fkey
    FOREIGN KEY (procedure_id) REFERENCES public.procedures(id)
);

CREATE INDEX IF NOT EXISTS idx_procedure_amendments_procedure ON public.procedure_amendments (procedure_id);

-- RLS
ALTER TABLE public.procedure_articles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.procedure_amendments ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Procedure articles are viewable by everyone"
  ON public.procedure_articles FOR SELECT USING (true);
CREATE POLICY "Service role can manage procedure articles"
  ON public.procedure_articles FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Procedure amendments are viewable by everyone"
  ON public.procedure_amendments FOR SELECT USING (true);
CREATE POLICY "Service role can manage procedure amendments"
  ON public.procedure_amendments FOR ALL USING (auth.role() = 'service_role');
