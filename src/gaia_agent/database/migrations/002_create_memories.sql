CREATE TABLE IF NOT EXISTS memories (
    memory_id UUID PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    memory_type VARCHAR(50) NOT NULL,
    source VARCHAR(50) NOT NULL,
    importance DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    metadata JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_memories_user_id
    ON memories(user_id);

CREATE INDEX IF NOT EXISTS idx_memories_user_active
    ON memories(user_id, active);

CREATE INDEX IF NOT EXISTS idx_memories_created_at
    ON memories(created_at);
    ALTER TABLE memories
ADD COLUMN IF NOT EXISTS last_accessed_at TIMESTAMPTZ;

ALTER TABLE memories
ADD COLUMN IF NOT EXISTS access_count INTEGER NOT NULL DEFAULT 0;