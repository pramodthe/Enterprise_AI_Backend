-- Supabase Database Schema for Enterprise AI Assistant Platform
-- This schema defines tables for users, sessions, and messages

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT UNIQUE NOT NULL,
    username TEXT NOT NULL,
    email TEXT,
    full_name TEXT,
    disabled BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index on user_id for faster lookups
CREATE INDEX IF NOT EXISTS idx_users_user_id ON users(user_id);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT UNIQUE NOT NULL,
    user_id TEXT REFERENCES users(user_id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    is_expired BOOLEAN DEFAULT FALSE,
    CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- Create indexes for sessions
CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_is_expired ON sessions(is_expired);
CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    agent_used TEXT,
    sequence_number INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT fk_session FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

-- Create indexes for messages
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);
CREATE INDEX IF NOT EXISTS idx_messages_sequence ON messages(session_id, sequence_number);

-- Agent metrics table (for analytics and monitoring)
CREATE TABLE IF NOT EXISTS agent_metrics (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT,
    agent_name TEXT NOT NULL,
    query TEXT NOT NULL,
    response_time_ms INTEGER,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    confidence_score FLOAT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT fk_session_metrics FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE SET NULL
);

-- Create indexes for agent metrics
CREATE INDEX IF NOT EXISTS idx_agent_metrics_session_id ON agent_metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_metrics_agent_name ON agent_metrics(agent_name);
CREATE INDEX IF NOT EXISTS idx_agent_metrics_created_at ON agent_metrics(created_at);
CREATE INDEX IF NOT EXISTS idx_agent_metrics_success ON agent_metrics(success);

-- Document metadata table (for RAG system)
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    document_id TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size INTEGER,
    uploaded_by TEXT,
    upload_date TIMESTAMPTZ DEFAULT NOW(),
    processed BOOLEAN DEFAULT FALSE,
    chunk_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb,
    CONSTRAINT fk_uploaded_by FOREIGN KEY (uploaded_by) REFERENCES users(user_id) ON DELETE SET NULL
);

-- Create indexes for documents
CREATE INDEX IF NOT EXISTS idx_documents_document_id ON documents(document_id);
CREATE INDEX IF NOT EXISTS idx_documents_uploaded_by ON documents(uploaded_by);
CREATE INDEX IF NOT EXISTS idx_documents_processed ON documents(processed);

-- Function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update updated_at on users table
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger to update updated_at on sessions table
CREATE TRIGGER update_sessions_updated_at
    BEFORE UPDATE ON sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Row Level Security (RLS) policies
-- Enable RLS on all tables
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Policy: Users can read their own data
CREATE POLICY users_select_own ON users
    FOR SELECT
    USING (auth.uid()::text = user_id);

-- Policy: Users can update their own data
CREATE POLICY users_update_own ON users
    FOR UPDATE
    USING (auth.uid()::text = user_id);

-- Policy: Users can read their own sessions
CREATE POLICY sessions_select_own ON sessions
    FOR SELECT
    USING (auth.uid()::text = user_id);

-- Policy: Users can insert their own sessions
CREATE POLICY sessions_insert_own ON sessions
    FOR INSERT
    WITH CHECK (auth.uid()::text = user_id);

-- Policy: Users can update their own sessions
CREATE POLICY sessions_update_own ON sessions
    FOR UPDATE
    USING (auth.uid()::text = user_id);

-- Policy: Users can delete their own sessions
CREATE POLICY sessions_delete_own ON sessions
    FOR DELETE
    USING (auth.uid()::text = user_id);

-- Policy: Users can read messages from their sessions
CREATE POLICY messages_select_own ON messages
    FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM sessions
            WHERE sessions.session_id = messages.session_id
            AND sessions.user_id = auth.uid()::text
        )
    );

-- Policy: Service role can do everything (for backend operations)
CREATE POLICY service_role_all_users ON users
    FOR ALL
    USING (auth.jwt()->>'role' = 'service_role');

CREATE POLICY service_role_all_sessions ON sessions
    FOR ALL
    USING (auth.jwt()->>'role' = 'service_role');

CREATE POLICY service_role_all_messages ON messages
    FOR ALL
    USING (auth.jwt()->>'role' = 'service_role');

CREATE POLICY service_role_all_metrics ON agent_metrics
    FOR ALL
    USING (auth.jwt()->>'role' = 'service_role');

CREATE POLICY service_role_all_documents ON documents
    FOR ALL
    USING (auth.jwt()->>'role' = 'service_role');

-- Views for analytics
CREATE OR REPLACE VIEW session_statistics AS
SELECT
    user_id,
    COUNT(*) as total_sessions,
    COUNT(CASE WHEN is_expired = FALSE THEN 1 END) as active_sessions,
    MAX(updated_at) as last_activity
FROM sessions
GROUP BY user_id;

CREATE OR REPLACE VIEW agent_performance AS
SELECT
    agent_name,
    COUNT(*) as total_queries,
    AVG(response_time_ms) as avg_response_time_ms,
    SUM(CASE WHEN success = TRUE THEN 1 ELSE 0 END)::float / COUNT(*) as success_rate,
    AVG(confidence_score) as avg_confidence
FROM agent_metrics
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY agent_name;

-- Comments for documentation
COMMENT ON TABLE users IS 'Stores user account information';
COMMENT ON TABLE sessions IS 'Stores conversation sessions with metadata';
COMMENT ON TABLE messages IS 'Stores individual messages within sessions';
COMMENT ON TABLE agent_metrics IS 'Stores performance metrics for AI agents';
COMMENT ON TABLE documents IS 'Stores metadata for uploaded documents';
