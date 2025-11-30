-- =============================================================================
-- AI Agent Performance Intelligence System - Database Initialization
-- Course: DATA 230 (Data Visualization) at SJSU
-- =============================================================================

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- Tables
-- =============================================================================

-- Prediction results table
CREATE TABLE IF NOT EXISTS prediction_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id VARCHAR(100) NOT NULL,
    endpoint VARCHAR(100) NOT NULL,
    request_payload JSONB NOT NULL,
    response_payload JSONB NOT NULL,
    model_version VARCHAR(50),
    latency_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent configurations table
CREATE TABLE IF NOT EXISTS agent_configurations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_type VARCHAR(100) NOT NULL,
    model_architecture VARCHAR(100) NOT NULL,
    deployment_environment VARCHAR(50) NOT NULL,
    autonomy_level INTEGER CHECK (autonomy_level >= 1 AND autonomy_level <= 10),
    expected_cost_per_task DECIMAL(10, 6),
    performance_score DECIMAL(5, 4),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Risk assessments table
CREATE TABLE IF NOT EXISTS risk_assessments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(100) NOT NULL,
    risk_score DECIMAL(5, 2),
    failure_probability DECIMAL(5, 4),
    risk_level VARCHAR(20),
    contributing_factors JSONB,
    mitigation_steps JSONB,
    assessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Recommendations history table
CREATE TABLE IF NOT EXISTS recommendations_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_profile JSONB NOT NULL,
    recommendations JSONB NOT NULL,
    best_match JSONB,
    confidence_score DECIMAL(5, 4),
    model_version VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- API metrics table
CREATE TABLE IF NOT EXISTS api_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    endpoint VARCHAR(100) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER,
    latency_ms INTEGER,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    user_agent VARCHAR(500),
    ip_address VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- Indexes
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_prediction_results_created_at ON prediction_results(created_at);
CREATE INDEX IF NOT EXISTS idx_prediction_results_endpoint ON prediction_results(endpoint);
CREATE INDEX IF NOT EXISTS idx_agent_configurations_type ON agent_configurations(agent_type);
CREATE INDEX IF NOT EXISTS idx_risk_assessments_agent_id ON risk_assessments(agent_id);
CREATE INDEX IF NOT EXISTS idx_api_metrics_endpoint ON api_metrics(endpoint);
CREATE INDEX IF NOT EXISTS idx_api_metrics_created_at ON api_metrics(created_at);

-- =============================================================================
-- Functions
-- =============================================================================

-- Update timestamp function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for agent_configurations
DROP TRIGGER IF EXISTS update_agent_configurations_updated_at ON agent_configurations;
CREATE TRIGGER update_agent_configurations_updated_at
    BEFORE UPDATE ON agent_configurations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Initial Data
-- =============================================================================

-- Insert sample agent configurations
INSERT INTO agent_configurations (agent_type, model_architecture, deployment_environment, autonomy_level, expected_cost_per_task, performance_score)
VALUES 
    ('Code Assistant', 'GPT-4-Turbo', 'Cloud', 7, 0.012, 0.87),
    ('Data Analyst', 'Claude-3-Opus', 'Hybrid', 6, 0.018, 0.82),
    ('Research Assistant', 'GPT-4', 'Cloud', 5, 0.015, 0.85),
    ('QA Tester', 'Mixtral-8x7B', 'Edge', 4, 0.008, 0.78),
    ('Content Writer', 'Claude-3-Sonnet', 'Cloud', 6, 0.010, 0.80)
ON CONFLICT DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

