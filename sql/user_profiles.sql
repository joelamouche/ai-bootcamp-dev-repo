CREATE SCHEMA IF NOT EXISTS user_profiles;

CREATE TABLE user_profiles.stored_user_profiles (
    id SERIAL PRIMARY KEY,
    handle VARCHAR(255) NOT NULL,
    display_name VARCHAR(255),
    description TEXT,
    did VARCHAR(255) NOT NULL UNIQUE,
    relevance_score DECIMAL(3, 2) NOT NULL DEFAULT 0.0,
    reasoning TEXT,
    follow_status VARCHAR(50) NOT NULL DEFAULT 'none',
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT valid_relevance_score CHECK (relevance_score >= 0.0 AND relevance_score <= 1.0),
    CONSTRAINT valid_follow_status CHECK (follow_status IN ('none', 'pending', 'followed', 'rejected', 'unfollowed'))
);

-- Index for faster queries by handle
CREATE INDEX idx_user_profiles_handle ON user_profiles.stored_user_profiles(handle);

-- Index for faster queries by did (already unique, but index helps with joins)
CREATE INDEX idx_user_profiles_did ON user_profiles.stored_user_profiles(did);

-- Index for faster queries by follow_status
CREATE INDEX idx_user_profiles_follow_status ON user_profiles.stored_user_profiles(follow_status);

-- Index for faster queries by relevance_score (for sorting)
CREATE INDEX idx_user_profiles_relevance_score ON user_profiles.stored_user_profiles(relevance_score DESC);

-- Trigger to automatically update the last_updated timestamp
CREATE OR REPLACE FUNCTION update_user_profile_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER user_profile_update_timestamp
    BEFORE UPDATE ON user_profiles.stored_user_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_user_profile_timestamp();

