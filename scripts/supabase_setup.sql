-- Supabase setup for OpenFOAM RAG
-- Run this in the Supabase SQL Editor

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    section TEXT DEFAULT 'N/A',
    subsection TEXT DEFAULT 'N/A',
    subsubsection TEXT DEFAULT 'N/A',
    page TEXT DEFAULT 'N/A',
    source TEXT NOT NULL,
    parser TEXT NOT NULL DEFAULT 'marker',
    word_count INTEGER,
    token_count INTEGER,
    embedding VECTOR(384) NOT NULL
);

CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);

CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding VECTOR(384),
    match_count INT DEFAULT 10,
    filter_parser TEXT DEFAULT 'marker'
)
RETURNS TABLE (
    id INT,
    content TEXT,
    section TEXT,
    subsection TEXT,
    subsubsection TEXT,
    page TEXT,
    source TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id,
        c.content,
        c.section,
        c.subsection,
        c.subsubsection,
        c.page,
        c.source,
        1 - (c.embedding <=> query_embedding) AS similarity
    FROM chunks c
    WHERE c.parser = filter_parser
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ---------------------------------------------------------------------------
-- Feedback table — anonymous thumbs up/down on answers
-- ---------------------------------------------------------------------------

CREATE TABLE feedback (
    id BIGSERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    answer_hash TEXT NOT NULL,
    answer_text TEXT DEFAULT NULL,
    mode TEXT NOT NULL DEFAULT 'quick',
    vote TEXT NOT NULL CHECK (vote IN ('up', 'down')),
    comment TEXT DEFAULT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_feedback_created_at ON feedback (created_at DESC);
