-- ============================================================
-- Migration: HYS feedback bronze tables
-- ============================================================
-- hys_feedback_bronze  : One row per feedback submission
-- hys_feedback_chunks  : Text chunks for keyword search / RAG
-- ============================================================

-- ----------------------------------------------------------------
-- 1. Raw feedback rows
-- ----------------------------------------------------------------
create table if not exists hys_feedback_bronze (
    -- Primary key: HYS internal feedback ID
    feedback_id             bigint          primary key,

    -- Initiative & procedure linkage
    initiative_id           integer         not null,
    procedure_id            text            not null
        references procedures(id) on delete cascade,
    com_number              text            not null,   -- normalised, e.g. "COM(2025)836"

    -- Respondent identity (the critical fields)
    user_type               text,                       -- ORGANISATION, BUSINESS_ASSOCIATION, etc.
    transparency_reg_id     text,                       -- EU Transparency Register number
    organisation_name       text,

    -- Geographic / linguistic metadata
    country                 text,
    language                text,

    -- Content
    feedback_text           text,                       -- inline text (rare; most use attachments)
    attachment_count        integer         default 0,
    pdf_extracted           boolean         default false, -- true if PDF text was successfully extracted

    -- Temporal & status
    date_feedback           timestamptz,
    publication_status      text,

    -- Full raw API response for re-parsing without re-scraping
    raw_json                jsonb           not null,

    -- Housekeeping
    scraped_at              timestamptz     default now()
);

-- Index for procedure-level lookups (most common query pattern)
create index if not exists hys_feedback_bronze_procedure_idx
    on hys_feedback_bronze (procedure_id);

-- Index for org linkage via Transparency Register
create index if not exists hys_feedback_bronze_tr_id_idx
    on hys_feedback_bronze (transparency_reg_id)
    where transparency_reg_id is not null;

-- Index for COM number search
create index if not exists hys_feedback_bronze_com_idx
    on hys_feedback_bronze (com_number);


-- ----------------------------------------------------------------
-- 2. Text chunks for keyword search / simplified RAG
-- ----------------------------------------------------------------
create table if not exists hys_feedback_chunks (
    id                      bigserial       primary key,

    -- Parent feedback linkage
    feedback_id             bigint          not null
        references hys_feedback_bronze(feedback_id) on delete cascade,
    initiative_id           integer         not null,
    procedure_id            text            not null,
    com_number              text            not null,

    -- Chunk position within this feedback
    chunk_index             integer         not null,
    chunk_total             integer         not null,

    -- The chunk text
    chunk_text              text            not null,

    -- Denormalised for faster keyword search without joins
    organisation_name       text,
    transparency_reg_id     text,
    date_feedback           timestamptz,

    -- Unique constraint to enable upsert
    unique (feedback_id, chunk_index)
);

-- Full-text search index (PostgreSQL tsvector)
create index if not exists hys_feedback_chunks_fts_idx
    on hys_feedback_chunks
    using gin (to_tsvector('english', chunk_text));

-- Index for feedback-level lookups
create index if not exists hys_feedback_chunks_feedback_idx
    on hys_feedback_chunks (feedback_id);

-- Index for procedure-level chunk queries
create index if not exists hys_feedback_chunks_procedure_idx
    on hys_feedback_chunks (procedure_id);

-- Index for TR-based org linkage on chunks
create index if not exists hys_feedback_chunks_tr_idx
    on hys_feedback_chunks (transparency_reg_id)
    where transparency_reg_id is not null;
