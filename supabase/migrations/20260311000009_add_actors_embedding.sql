-- Add embedding column to actors table (requires pgvector extension)
CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE public.actors
  ADD COLUMN IF NOT EXISTS embedding vector(1536);
