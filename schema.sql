-- =====================================================================
-- EU Lobbying Influence — Postgres schema snapshot
-- =====================================================================
-- Reproduces the database schema (public namespace only) as it stood
-- at the time of the thesis hand-in. Apply to a fresh Postgres/Supabase
-- database with:
--
--     psql "$DATABASE_URL" -f supabase/schema.sql
--
-- =====================================================================


-- =====================================================================
-- 1. Extensions
-- =====================================================================
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;


-- =====================================================================
-- 2. Sequences
-- =====================================================================
CREATE SEQUENCE IF NOT EXISTS public.hys_feedback_chunks_id_seq
  AS bigint START 1 INCREMENT 1 MINVALUE 1 MAXVALUE 9223372036854775807 CACHE 1;


-- =====================================================================
-- 3. Tables (columns only; constraints/indexes follow)
-- =====================================================================

CREATE TABLE public.actors (
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
  role_summary text,
  key_topics text[],
  declarations_summary text,
  embedding_model text,
  parliament text DEFAULT 'eu'::text,
  description text,
  status text DEFAULT 'active'::text,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  embedding vector
);

CREATE TABLE public.commission_meeting_organizations (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  meeting_id text NOT NULL,
  organization_id text,
  organization_name text NOT NULL,
  eu_transparency_register_id text,
  org_match_method text
);

CREATE TABLE public.commission_meetings (
  id text NOT NULL,
  commissioner_name text NOT NULL,
  commissioner_portfolio text,
  host_id text,
  meeting_type text DEFAULT 'commissioner'::text,
  meeting_date date,
  location text,
  subject text,
  commission_representatives jsonb DEFAULT '[]'::jsonb,
  organizations_raw text,
  transparency_register_ids text[] DEFAULT '{}'::text[],
  points_raised text,
  conclusions text,
  ares_number text,
  minutes_url text,
  source_url text,
  raw_data jsonb,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  actor_id text,
  match_status text
);

CREATE TABLE public.hys_feedback_bronze (
  feedback_id bigint NOT NULL,
  initiative_id integer NOT NULL,
  procedure_id text NOT NULL,
  com_number text NOT NULL,
  user_type text,
  transparency_reg_id text,
  organisation_name text,
  country text,
  language text,
  feedback_text text,
  attachment_count integer DEFAULT 0,
  pdf_extracted boolean DEFAULT false,
  date_feedback timestamp with time zone,
  publication_status text,
  raw_json jsonb NOT NULL,
  scraped_at timestamp with time zone DEFAULT now()
);

CREATE TABLE public.hys_feedback_chunks (
  id bigint NOT NULL DEFAULT nextval('public.hys_feedback_chunks_id_seq'::regclass),
  feedback_id bigint NOT NULL,
  initiative_id integer NOT NULL,
  procedure_id text NOT NULL,
  com_number text NOT NULL,
  chunk_index integer NOT NULL,
  chunk_total integer NOT NULL,
  chunk_text text NOT NULL,
  organisation_name text,
  transparency_reg_id text,
  date_feedback timestamp with time zone
);

CREATE TABLE public.lobbying_meetings (
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
  match_status text,
  org_match_method text
);

CREATE TABLE public.meeting_procedure_links (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  lobbying_meeting_id text,
  commission_meeting_id text,
  procedure_id text NOT NULL,
  match_method text NOT NULL,
  match_rank integer DEFAULT 1,
  is_primary boolean DEFAULT false,
  match_details jsonb,
  created_at timestamp with time zone DEFAULT now()
);

CREATE TABLE public.meps (
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
  declarations_summary text,
  speech_summary text DEFAULT ''::text,
  speech_top_words jsonb DEFAULT '[]'::jsonb,
  speech_sources jsonb DEFAULT '[]'::jsonb,
  status text DEFAULT 'active'::text,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now()
);

CREATE TABLE public.organizations (
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
  dedup_status text,
  match_method text,
  matched_tr_id text
);

CREATE TABLE public.procedure_aliases (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  procedure_id text NOT NULL,
  alias text NOT NULL,
  alias_type text,
  created_at timestamp with time zone DEFAULT now()
);

CREATE TABLE public.procedure_amendments (
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
  created_at timestamp with time zone DEFAULT now()
);

CREATE TABLE public.procedure_articles (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  procedure_id text NOT NULL,
  element_type text NOT NULL,
  element_number text NOT NULL,
  title text,
  content text NOT NULL,
  document_source text NOT NULL,
  document_version text NOT NULL,
  sort_order integer DEFAULT 0,
  created_at timestamp with time zone DEFAULT now()
);

CREATE TABLE public.procedure_documents (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  procedure_id text NOT NULL,
  document_id text NOT NULL,
  document_type text NOT NULL,
  committee text,
  rapporteur text,
  title text,
  url text,
  pdf_url text,
  content_text text,
  event_date date,
  page_count integer,
  file_size_bytes integer,
  created_at timestamp with time zone DEFAULT now()
);

CREATE TABLE public.procedure_texts (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  procedure_id text NOT NULL,
  proposal_id text,
  proposal_text text,
  final_act_id text,
  final_act_text text,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now()
);

CREATE TABLE public.procedures (
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
  ai_summary text,
  ai_impact_analysis jsonb,
  ai_next_steps text,
  embedding_model text,
  api_uri text,
  oeil_url text,
  eurlex_proposal_url text,
  eurlex_final_act_url text,
  is_deleted boolean DEFAULT false,
  deleted_at timestamp with time zone,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  responsible_committee text,
  rapporteurs jsonb DEFAULT '[]'::jsonb,
  shadow_rapporteurs jsonb DEFAULT '[]'::jsonb,
  rapporteurs_for_opinion jsonb DEFAULT '[]'::jsonb,
  commission_dg text,
  commissioner text,
  amendments_tabled_date date,
  amendment_vote_date date,
  regulation_vote_date date,
  date_of_final_act_signed date
);


-- =====================================================================
-- 4. Primary keys
-- =====================================================================
ALTER TABLE public.actors                           ADD CONSTRAINT actors_pkey                            PRIMARY KEY (actor_id);
ALTER TABLE public.commission_meeting_organizations ADD CONSTRAINT commission_meeting_organizations_pkey PRIMARY KEY (id);
ALTER TABLE public.commission_meetings              ADD CONSTRAINT commission_meetings_pkey              PRIMARY KEY (id);
ALTER TABLE public.hys_feedback_bronze              ADD CONSTRAINT hys_feedback_bronze_pkey              PRIMARY KEY (feedback_id);
ALTER TABLE public.hys_feedback_chunks              ADD CONSTRAINT hys_feedback_chunks_pkey              PRIMARY KEY (id);
ALTER TABLE public.lobbying_meetings                ADD CONSTRAINT lobbying_meetings_pkey                PRIMARY KEY (id);
ALTER TABLE public.meeting_procedure_links          ADD CONSTRAINT meeting_procedure_links_pkey          PRIMARY KEY (id);
ALTER TABLE public.meps                             ADD CONSTRAINT meps_pkey                             PRIMARY KEY (id);
ALTER TABLE public.organizations                    ADD CONSTRAINT organizations_pkey                    PRIMARY KEY (id);
ALTER TABLE public.procedure_aliases                ADD CONSTRAINT procedure_aliases_pkey                PRIMARY KEY (id);
ALTER TABLE public.procedure_amendments             ADD CONSTRAINT procedure_amendments_pkey             PRIMARY KEY (id);
ALTER TABLE public.procedure_articles               ADD CONSTRAINT procedure_articles_pkey               PRIMARY KEY (id);
ALTER TABLE public.procedure_documents              ADD CONSTRAINT procedure_documents_pkey              PRIMARY KEY (id);
ALTER TABLE public.procedure_texts                  ADD CONSTRAINT procedure_texts_pkey                  PRIMARY KEY (id);
ALTER TABLE public.procedures                       ADD CONSTRAINT procedures_pkey                       PRIMARY KEY (id);


-- =====================================================================
-- 5. Unique constraints
-- =====================================================================
ALTER TABLE public.hys_feedback_chunks  ADD CONSTRAINT hys_feedback_chunks_feedback_id_chunk_index_key UNIQUE (feedback_id, chunk_index);
ALTER TABLE public.procedure_aliases    ADD CONSTRAINT procedure_aliases_unique                        UNIQUE (alias, procedure_id);
ALTER TABLE public.procedure_documents  ADD CONSTRAINT procedure_documents_procedure_id_document_id_key UNIQUE (procedure_id, document_id);
ALTER TABLE public.procedure_texts      ADD CONSTRAINT procedure_texts_procedure_id_key                 UNIQUE (procedure_id);


-- =====================================================================
-- 6. Check constraints
-- =====================================================================
ALTER TABLE public.actors                  ADD CONSTRAINT actors_status_check CHECK (status = ANY (ARRAY['active'::text, 'inactive'::text]));
ALTER TABLE public.meps                    ADD CONSTRAINT meps_status_check   CHECK (status = ANY (ARRAY['active'::text, 'inactive'::text]));
ALTER TABLE public.meeting_procedure_links ADD CONSTRAINT meeting_procedure_links_one_source_check
  CHECK (((lobbying_meeting_id IS NOT NULL) AND (commission_meeting_id IS NULL))
      OR ((lobbying_meeting_id IS NULL) AND (commission_meeting_id IS NOT NULL)));
ALTER TABLE public.procedure_articles      ADD CONSTRAINT procedure_articles_document_version_check
  CHECK (document_version = ANY (ARRAY['proposal'::text, 'committee'::text, 'adopted'::text]));
ALTER TABLE public.procedure_articles      ADD CONSTRAINT procedure_articles_element_type_check
  CHECK (element_type = ANY (ARRAY['recital'::text, 'article'::text]));


-- =====================================================================
-- 7. Foreign keys (all tables now exist, so refs can be created)
-- =====================================================================
ALTER TABLE public.commission_meeting_organizations
  ADD CONSTRAINT commission_meeting_organizations_meeting_fkey FOREIGN KEY (meeting_id)      REFERENCES public.commission_meetings(id) ON DELETE CASCADE,
  ADD CONSTRAINT commission_meeting_organizations_org_fkey     FOREIGN KEY (organization_id) REFERENCES public.organizations(id)        ON DELETE SET NULL;

ALTER TABLE public.commission_meetings
  ADD CONSTRAINT commission_meetings_actor_fkey FOREIGN KEY (actor_id) REFERENCES public.actors(actor_id) ON DELETE SET NULL;

ALTER TABLE public.hys_feedback_bronze
  ADD CONSTRAINT hys_feedback_bronze_procedure_id_fkey FOREIGN KEY (procedure_id) REFERENCES public.procedures(id) ON DELETE CASCADE;

ALTER TABLE public.hys_feedback_chunks
  ADD CONSTRAINT hys_feedback_chunks_feedback_id_fkey FOREIGN KEY (feedback_id) REFERENCES public.hys_feedback_bronze(feedback_id) ON DELETE CASCADE;

ALTER TABLE public.lobbying_meetings
  ADD CONSTRAINT lobbying_meetings_mep_id_fkey          FOREIGN KEY (mep_id)          REFERENCES public.meps(id),
  ADD CONSTRAINT lobbying_meetings_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organizations(id);

ALTER TABLE public.meeting_procedure_links
  ADD CONSTRAINT meeting_procedure_links_commission_fkey FOREIGN KEY (commission_meeting_id) REFERENCES public.commission_meetings(id),
  ADD CONSTRAINT meeting_procedure_links_lobbying_fkey   FOREIGN KEY (lobbying_meeting_id)   REFERENCES public.lobbying_meetings(id),
  ADD CONSTRAINT meeting_procedure_links_procedure_fkey  FOREIGN KEY (procedure_id)          REFERENCES public.procedures(id);

ALTER TABLE public.procedure_aliases
  ADD CONSTRAINT procedure_aliases_procedure_fkey FOREIGN KEY (procedure_id) REFERENCES public.procedures(id) ON DELETE CASCADE;

ALTER TABLE public.procedure_amendments
  ADD CONSTRAINT procedure_amendments_procedure_id_fkey FOREIGN KEY (procedure_id) REFERENCES public.procedures(id);

ALTER TABLE public.procedure_articles
  ADD CONSTRAINT procedure_articles_procedure_id_fkey FOREIGN KEY (procedure_id) REFERENCES public.procedures(id);

ALTER TABLE public.procedure_documents
  ADD CONSTRAINT procedure_documents_procedure_id_fkey FOREIGN KEY (procedure_id) REFERENCES public.procedures(id);

ALTER TABLE public.procedure_texts
  ADD CONSTRAINT procedure_texts_procedure_id_fkey FOREIGN KEY (procedure_id) REFERENCES public.procedures(id);


-- =====================================================================
-- 8. Indexes (non-implicit; PK/UNIQUE indexes were created above)
-- =====================================================================
CREATE INDEX idx_commission_meeting_orgs_meeting          ON public.commission_meeting_organizations USING btree (meeting_id);
CREATE INDEX idx_commission_meeting_orgs_org              ON public.commission_meeting_organizations USING btree (organization_id);
CREATE INDEX idx_commission_meeting_orgs_org_match_method ON public.commission_meeting_organizations USING btree (org_match_method);
CREATE INDEX idx_commission_meeting_orgs_tr               ON public.commission_meeting_organizations USING btree (eu_transparency_register_id);
CREATE INDEX idx_commission_meetings_actor                ON public.commission_meetings USING btree (actor_id);
CREATE INDEX idx_commission_meetings_commissioner         ON public.commission_meetings USING btree (commissioner_name);
CREATE INDEX idx_commission_meetings_date                 ON public.commission_meetings USING btree (meeting_date);
CREATE INDEX idx_commission_meetings_host                 ON public.commission_meetings USING btree (host_id);
CREATE INDEX idx_commission_meetings_match_status         ON public.commission_meetings USING btree (match_status) WHERE (match_status IS NULL);

CREATE INDEX hys_feedback_bronze_com_idx       ON public.hys_feedback_bronze USING btree (com_number);
CREATE INDEX hys_feedback_bronze_procedure_idx ON public.hys_feedback_bronze USING btree (procedure_id);
CREATE INDEX hys_feedback_bronze_tr_id_idx     ON public.hys_feedback_bronze USING btree (transparency_reg_id) WHERE (transparency_reg_id IS NOT NULL);
CREATE INDEX hys_feedback_chunks_feedback_idx  ON public.hys_feedback_chunks USING btree (feedback_id);
CREATE INDEX hys_feedback_chunks_fts_idx       ON public.hys_feedback_chunks USING gin (to_tsvector('english'::regconfig, chunk_text));
CREATE INDEX hys_feedback_chunks_procedure_idx ON public.hys_feedback_chunks USING btree (procedure_id);
CREATE INDEX hys_feedback_chunks_tr_idx        ON public.hys_feedback_chunks USING btree (transparency_reg_id) WHERE (transparency_reg_id IS NOT NULL);

CREATE INDEX idx_lobbying_meetings_committee        ON public.lobbying_meetings USING btree (committee_acronym);
CREATE INDEX idx_lobbying_meetings_date             ON public.lobbying_meetings USING btree (meeting_date);
CREATE INDEX idx_lobbying_meetings_match_status     ON public.lobbying_meetings USING btree (match_status) WHERE (match_status IS NULL);
CREATE INDEX idx_lobbying_meetings_mep_id           ON public.lobbying_meetings USING btree (mep_id);
CREATE INDEX idx_lobbying_meetings_org_id           ON public.lobbying_meetings USING btree (organization_id);
CREATE INDEX idx_lobbying_meetings_org_match_method ON public.lobbying_meetings USING btree (org_match_method);

CREATE INDEX idx_mpl_commission ON public.meeting_procedure_links USING btree (commission_meeting_id) WHERE (commission_meeting_id IS NOT NULL);
CREATE INDEX idx_mpl_lobbying   ON public.meeting_procedure_links USING btree (lobbying_meeting_id)   WHERE (lobbying_meeting_id IS NOT NULL);
CREATE INDEX idx_mpl_method     ON public.meeting_procedure_links USING btree (match_method);
CREATE INDEX idx_mpl_procedure  ON public.meeting_procedure_links USING btree (procedure_id);

CREATE INDEX idx_meps_country         ON public.meps USING btree (country);
CREATE INDEX idx_meps_political_group ON public.meps USING btree ("politicalGroup");
CREATE INDEX idx_meps_status          ON public.meps USING btree (status);

CREATE INDEX idx_organizations_country         ON public.organizations USING btree (country);
CREATE INDEX idx_organizations_dedup_status    ON public.organizations USING btree (dedup_status) WHERE ((dedup_status IS NULL) AND (eu_transparency_register_id IS NULL));
CREATE INDEX idx_organizations_match_method    ON public.organizations USING btree (match_method) WHERE (match_method IS NULL);
CREATE INDEX idx_organizations_name            ON public.organizations USING btree (name);
CREATE INDEX idx_organizations_name_trgm       ON public.organizations USING gin (name gin_trgm_ops);
CREATE INDEX idx_organizations_transparency_id ON public.organizations USING btree (eu_transparency_register_id);
CREATE INDEX idx_organizations_type            ON public.organizations USING btree (organization_type);

CREATE INDEX idx_procedure_aliases_alias     ON public.procedure_aliases USING btree (lower(alias));
CREATE INDEX idx_procedure_aliases_procedure ON public.procedure_aliases USING btree (procedure_id);

CREATE INDEX idx_procedure_amendments_procedure ON public.procedure_amendments USING btree (procedure_id);

CREATE INDEX idx_procedure_articles_procedure ON public.procedure_articles USING btree (procedure_id);
CREATE INDEX idx_procedure_articles_type      ON public.procedure_articles USING btree (element_type);

CREATE INDEX idx_proc_docs_procedure ON public.procedure_documents USING btree (procedure_id);
CREATE INDEX idx_proc_docs_type      ON public.procedure_documents USING btree (document_type);

CREATE INDEX idx_procedure_texts_procedure ON public.procedure_texts USING btree (procedure_id);

CREATE INDEX idx_procedures_policy_area   ON public.procedures USING btree (policy_area);
CREATE INDEX idx_procedures_process_id    ON public.procedures USING btree (process_id);
CREATE INDEX idx_procedures_proposal_date ON public.procedures USING btree (proposal_date);
CREATE INDEX idx_procedures_status        ON public.procedures USING btree (status);
CREATE INDEX idx_procedures_subjects      ON public.procedures USING gin (subjects);
CREATE INDEX idx_procedures_title_trgm    ON public.procedures USING gin (title gin_trgm_ops);
CREATE INDEX idx_procedures_type          ON public.procedures USING btree (procedure_type);


-- =====================================================================
-- 9. Functions
-- =====================================================================

CREATE OR REPLACE FUNCTION public.update_updated_at_column()
  RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION public.cleanup_orphaned_stubs()
  RETURNS integer LANGUAGE plpgsql SECURITY DEFINER AS $$
DECLARE
  deleted_count integer;
BEGIN
  DELETE FROM public.organizations o
  WHERE o.eu_transparency_register_id IS NULL
    AND NOT EXISTS (SELECT 1 FROM public.lobbying_meetings lm WHERE lm.organization_id = o.id)
    AND NOT EXISTS (SELECT 1 FROM public.commission_meeting_organizations cmo WHERE cmo.organization_id = o.id);
  GET DIAGNOSTICS deleted_count = ROW_COUNT;
  RETURN deleted_count;
END;
$$;

CREATE OR REPLACE FUNCTION public.rls_auto_enable()
  RETURNS event_trigger LANGUAGE plpgsql SECURITY DEFINER
  SET search_path TO 'pg_catalog' AS $$
DECLARE
  cmd record;
BEGIN
  FOR cmd IN
    SELECT *
    FROM pg_event_trigger_ddl_commands()
    WHERE command_tag IN ('CREATE TABLE', 'CREATE TABLE AS', 'SELECT INTO')
      AND object_type IN ('table', 'partitioned table')
  LOOP
    IF cmd.schema_name = 'public' THEN
      BEGIN
        EXECUTE format('alter table if exists %s enable row level security', cmd.object_identity);
        RAISE LOG 'rls_auto_enable: enabled RLS on %', cmd.object_identity;
      EXCEPTION WHEN OTHERS THEN
        RAISE LOG 'rls_auto_enable: failed to enable RLS on %', cmd.object_identity;
      END;
    END IF;
  END LOOP;
END;
$$;


-- =====================================================================
-- 10. Triggers (updated_at columns auto-set on UPDATE)
-- =====================================================================
CREATE TRIGGER actors_updated_at              BEFORE UPDATE ON public.actors              FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
CREATE TRIGGER commission_meetings_updated_at BEFORE UPDATE ON public.commission_meetings FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
CREATE TRIGGER meps_updated_at                BEFORE UPDATE ON public.meps                FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
CREATE TRIGGER organizations_updated_at       BEFORE UPDATE ON public.organizations       FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
CREATE TRIGGER procedures_updated_at          BEFORE UPDATE ON public.procedures          FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


-- =====================================================================
-- 11. Event trigger (auto-enables RLS on any newly-created public table)
-- =====================================================================
CREATE EVENT TRIGGER ensure_rls ON ddl_command_end
  WHEN TAG IN ('CREATE TABLE', 'CREATE TABLE AS', 'SELECT INTO')
  EXECUTE FUNCTION public.rls_auto_enable();


-- =====================================================================
-- 12. Row-Level Security
--   * SELECT is open to anyone (anonymous + authenticated)
--   * Writes (INSERT/UPDATE/DELETE) are restricted to the Supabase service_role
--   * Tables with RLS enabled but no policy default-deny all access
-- =====================================================================
ALTER TABLE public.actors                           ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.commission_meeting_organizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.commission_meetings              ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.hys_feedback_bronze              ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.hys_feedback_chunks              ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.lobbying_meetings                ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.meeting_procedure_links          ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.meps                             ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.organizations                    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.procedure_aliases                ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.procedure_amendments             ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.procedure_articles               ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.procedure_documents              ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.procedure_texts                  ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.procedures                       ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Actors are viewable by everyone" ON public.actors FOR SELECT TO public USING (true);
CREATE POLICY "Service role can manage actors"  ON public.actors FOR ALL    TO public USING (auth.role() = 'service_role'::text);

CREATE POLICY "Commission meeting organizations are viewable by everyone" ON public.commission_meeting_organizations FOR SELECT TO public USING (true);
CREATE POLICY "Service role can manage commission meeting organizations"  ON public.commission_meeting_organizations FOR ALL    TO public USING (auth.role() = 'service_role'::text);

CREATE POLICY "Commission meetings are viewable by everyone" ON public.commission_meetings FOR SELECT TO public USING (true);
CREATE POLICY "Service role can manage commission meetings"  ON public.commission_meetings FOR ALL    TO public USING (auth.role() = 'service_role'::text);

CREATE POLICY "Lobbying meetings are viewable by everyone" ON public.lobbying_meetings FOR SELECT TO public USING (true);
CREATE POLICY "Service role can manage lobbying meetings"  ON public.lobbying_meetings FOR ALL    TO public USING (auth.role() = 'service_role'::text);

CREATE POLICY "MEPs are viewable by everyone" ON public.meps FOR SELECT TO public USING (true);
CREATE POLICY "Service role can manage MEPs"  ON public.meps FOR ALL    TO public USING (auth.role() = 'service_role'::text);

CREATE POLICY "Organizations are viewable by everyone" ON public.organizations FOR SELECT TO public USING (true);
CREATE POLICY "Service role can manage organizations"  ON public.organizations FOR ALL    TO public USING (auth.role() = 'service_role'::text);

CREATE POLICY "Procedure aliases are viewable by everyone" ON public.procedure_aliases FOR SELECT TO public USING (true);
CREATE POLICY "Service role can manage procedure aliases"  ON public.procedure_aliases FOR ALL    TO public USING (auth.role() = 'service_role'::text);

CREATE POLICY "Procedure amendments are viewable by everyone" ON public.procedure_amendments FOR SELECT TO public USING (true);
CREATE POLICY "Service role can manage procedure amendments"  ON public.procedure_amendments FOR ALL    TO public USING (auth.role() = 'service_role'::text);

CREATE POLICY "Procedure articles are viewable by everyone" ON public.procedure_articles FOR SELECT TO public USING (true);
CREATE POLICY "Service role can manage procedure articles"  ON public.procedure_articles FOR ALL    TO public USING (auth.role() = 'service_role'::text);

CREATE POLICY "Procedure texts are viewable by everyone" ON public.procedure_texts FOR SELECT TO public USING (true);
CREATE POLICY "Service role can manage procedure texts"  ON public.procedure_texts FOR ALL    TO public USING (auth.role() = 'service_role'::text);

CREATE POLICY "Procedures are viewable by everyone" ON public.procedures FOR SELECT TO public USING (true);
CREATE POLICY "Service role can manage procedures"  ON public.procedures FOR ALL    TO public USING (auth.role() = 'service_role'::text);
