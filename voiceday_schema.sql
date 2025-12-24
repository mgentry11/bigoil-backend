-- VoiceDay Supabase Schema
-- Run this in the Supabase SQL Editor

-- =============================================
-- VOICEDAY USERS TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS voiceday_users (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    device_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL DEFAULT 'Anonymous',
    phone TEXT,
    push_token TEXT,
    personality TEXT DEFAULT 'Dr. Pemberton-Finch',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for phone lookups
CREATE INDEX IF NOT EXISTS idx_voiceday_users_phone ON voiceday_users(phone);
CREATE INDEX IF NOT EXISTS idx_voiceday_users_device_id ON voiceday_users(device_id);

-- =============================================
-- VOICEDAY CONNECTIONS TABLE
-- (Family/Friends relationships)
-- =============================================
CREATE TABLE IF NOT EXISTS voiceday_connections (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    owner_device_id TEXT NOT NULL,
    connected_device_id TEXT,  -- NULL if they don't have the app yet
    connected_phone TEXT NOT NULL,
    relationship TEXT DEFAULT 'Friend',
    nickname TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Foreign key to users (optional, since connected user might not exist yet)
    CONSTRAINT fk_owner FOREIGN KEY (owner_device_id)
        REFERENCES voiceday_users(device_id) ON DELETE CASCADE
);

-- Indexes for connection queries
CREATE INDEX IF NOT EXISTS idx_voiceday_connections_owner ON voiceday_connections(owner_device_id);
CREATE INDEX IF NOT EXISTS idx_voiceday_connections_connected ON voiceday_connections(connected_device_id);
CREATE INDEX IF NOT EXISTS idx_voiceday_connections_phone ON voiceday_connections(connected_phone);

-- =============================================
-- VOICEDAY SHARED TASKS TABLE
-- (Tasks assigned to family/friends)
-- =============================================
CREATE TABLE IF NOT EXISTS voiceday_shared_tasks (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    owner_device_id TEXT NOT NULL,
    assigned_device_id TEXT,  -- NULL if assignee doesn't have app
    assigned_phone TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    deadline TIMESTAMPTZ,
    priority TEXT DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high')),
    nag_interval_minutes INTEGER DEFAULT 15,
    is_completed BOOLEAN DEFAULT FALSE,
    completed_at TIMESTAMPTZ,
    last_nag_at TIMESTAMPTZ,
    nag_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT fk_task_owner FOREIGN KEY (owner_device_id)
        REFERENCES voiceday_users(device_id) ON DELETE CASCADE
);

-- Indexes for task queries
CREATE INDEX IF NOT EXISTS idx_voiceday_tasks_owner ON voiceday_shared_tasks(owner_device_id);
CREATE INDEX IF NOT EXISTS idx_voiceday_tasks_assigned ON voiceday_shared_tasks(assigned_device_id);
CREATE INDEX IF NOT EXISTS idx_voiceday_tasks_phone ON voiceday_shared_tasks(assigned_phone);
CREATE INDEX IF NOT EXISTS idx_voiceday_tasks_incomplete ON voiceday_shared_tasks(is_completed) WHERE is_completed = FALSE;

-- =============================================
-- VOICEDAY NAGS TABLE
-- (Nag messages sent between users)
-- =============================================
CREATE TABLE IF NOT EXISTS voiceday_nags (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    from_device_id TEXT NOT NULL,
    to_device_id TEXT,  -- NULL if recipient doesn't have app
    to_phone TEXT,
    task_id UUID REFERENCES voiceday_shared_tasks(id) ON DELETE SET NULL,
    message TEXT NOT NULL,
    delivery_method TEXT DEFAULT 'app' CHECK (delivery_method IN ('app', 'sms', 'push')),
    delivered_at TIMESTAMPTZ,
    acknowledged_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT fk_nag_sender FOREIGN KEY (from_device_id)
        REFERENCES voiceday_users(device_id) ON DELETE CASCADE
);

-- Indexes for nag queries
CREATE INDEX IF NOT EXISTS idx_voiceday_nags_from ON voiceday_nags(from_device_id);
CREATE INDEX IF NOT EXISTS idx_voiceday_nags_to ON voiceday_nags(to_device_id);
CREATE INDEX IF NOT EXISTS idx_voiceday_nags_task ON voiceday_nags(task_id);
CREATE INDEX IF NOT EXISTS idx_voiceday_nags_unacknowledged ON voiceday_nags(acknowledged_at) WHERE acknowledged_at IS NULL;

-- =============================================
-- UPDATED_AT TRIGGER FUNCTION
-- =============================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to tables with updated_at
DROP TRIGGER IF EXISTS update_voiceday_users_updated_at ON voiceday_users;
CREATE TRIGGER update_voiceday_users_updated_at
    BEFORE UPDATE ON voiceday_users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_voiceday_tasks_updated_at ON voiceday_shared_tasks;
CREATE TRIGGER update_voiceday_tasks_updated_at
    BEFORE UPDATE ON voiceday_shared_tasks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================
-- ROW LEVEL SECURITY (RLS)
-- Enable for production security
-- =============================================

-- For now, allow all access (you can tighten this later)
ALTER TABLE voiceday_users ENABLE ROW LEVEL SECURITY;
ALTER TABLE voiceday_connections ENABLE ROW LEVEL SECURITY;
ALTER TABLE voiceday_shared_tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE voiceday_nags ENABLE ROW LEVEL SECURITY;

-- Policies for service role access (backend uses service key)
CREATE POLICY "Service role full access on voiceday_users" ON voiceday_users
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Service role full access on voiceday_connections" ON voiceday_connections
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Service role full access on voiceday_shared_tasks" ON voiceday_shared_tasks
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Service role full access on voiceday_nags" ON voiceday_nags
    FOR ALL USING (true) WITH CHECK (true);

-- =============================================
-- SAMPLE DATA (Optional - for testing)
-- =============================================

-- Uncomment to insert test data:
/*
INSERT INTO voiceday_users (device_id, name, phone, personality) VALUES
    ('test-device-001', 'Test Parent', '+1234567890', 'Dr. Pemberton-Finch'),
    ('test-device-002', 'Test Child', '+0987654321', 'Sergeant Focus');

INSERT INTO voiceday_connections (owner_device_id, connected_device_id, connected_phone, relationship, nickname) VALUES
    ('test-device-001', 'test-device-002', '+0987654321', 'Child', 'Junior');

INSERT INTO voiceday_shared_tasks (owner_device_id, assigned_device_id, assigned_phone, title, priority, nag_interval_minutes) VALUES
    ('test-device-001', 'test-device-002', '+0987654321', 'Clean your room', 'high', 10),
    ('test-device-001', 'test-device-002', '+0987654321', 'Do homework', 'medium', 15);
*/

-- =============================================
-- VERIFICATION QUERIES
-- =============================================

-- Check tables were created:
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public' AND table_name LIKE 'voiceday_%';

-- Show table structures:
-- \d voiceday_users
-- \d voiceday_connections
-- \d voiceday_shared_tasks
-- \d voiceday_nags
