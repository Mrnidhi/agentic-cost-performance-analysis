/**
 * Risk Assessment Dashboard Component
 * Course: DATA 230 (Data Visualization) at SJSU
 * 
 * Real-time risk monitoring and assessment for AI agents.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { StrategicMLClient } from '../js/StrategicMLClient';

const apiClient = new StrategicMLClient(process.env.REACT_APP_API_URL || 'http://localhost:8000');

// Risk level colors
const RISK_COLORS = {
  Low: '#4caf50',
  Medium: '#ff9800',
  High: '#f44336',
  Critical: '#9c27b0'
};

// Sample agents for demo
const SAMPLE_AGENTS = [
  { agent_id: 'AG_001', name: 'Code Assistant Alpha' },
  { agent_id: 'AG_002', name: 'Data Analyst Beta' },
  { agent_id: 'AG_003', name: 'Research Agent Gamma' },
  { agent_id: 'AG_004', name: 'QA Tester Delta' }
];

const RiskAssessmentDashboard = () => {
  // State
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [agentState, setAgentState] = useState({
    agent_id: '',
    success_rate: 0.75,
    accuracy_score: 0.80,
    efficiency_score: 0.70,
    execution_time_seconds: 5.0,
    response_latency_ms: 300,
    memory_usage_mb: 350,
    cpu_usage_percent: 60,
    error_recovery_rate: 0.65
  });
  const [riskData, setRiskData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(false);

  // Assess risk
  const assessRisk = useCallback(async () => {
    if (!agentState.agent_id) return;
    
    setLoading(true);
    setError(null);

    try {
      const response = await apiClient.getFailureRisk(agentState);
      setRiskData(response);
    } catch (err) {
      setError(err.message || 'Failed to assess risk');
    } finally {
      setLoading(false);
    }
  }, [agentState]);

  // Auto-refresh effect
  useEffect(() => {
    if (!autoRefresh || !agentState.agent_id) return;

    const interval = setInterval(() => {
      // Simulate metric changes
      setAgentState(prev => ({
        ...prev,
        success_rate: Math.max(0.3, Math.min(1, prev.success_rate + (Math.random() - 0.5) * 0.05)),
        cpu_usage_percent: Math.max(20, Math.min(95, prev.cpu_usage_percent + (Math.random() - 0.5) * 10)),
        response_latency_ms: Math.max(100, Math.min(1000, prev.response_latency_ms + (Math.random() - 0.5) * 50))
      }));
      assessRisk();
    }, 5000);

    return () => clearInterval(interval);
  }, [autoRefresh, agentState.agent_id, assessRisk]);

  // Select agent
  const handleSelectAgent = (agent) => {
    setSelectedAgent(agent);
    setAgentState(prev => ({ ...prev, agent_id: agent.agent_id }));
    setRiskData(null);
  };

  // Update metric
  const updateMetric = (field, value) => {
    setAgentState(prev => ({ ...prev, [field]: value }));
  };

  // Get risk gauge rotation
  const getGaugeRotation = (score) => {
    return (score / 100) * 180 - 90;
  };

  return (
    <div className="risk-dashboard">
      <header className="dashboard-header">
        <h1>🛡️ Risk Assessment Dashboard</h1>
        <div className="header-controls">
          <label className="auto-refresh">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto-refresh (5s)
          </label>
        </div>
      </header>

      <div className="dashboard-grid">
        {/* Agent Selection */}
        <div className="panel agent-selection">
          <h2>Select Agent</h2>
          <div className="agent-list">
            {SAMPLE_AGENTS.map(agent => (
              <button
                key={agent.agent_id}
                className={`agent-btn ${selectedAgent?.agent_id === agent.agent_id ? 'active' : ''}`}
                onClick={() => handleSelectAgent(agent)}
              >
                <span className="agent-id">{agent.agent_id}</span>
                <span className="agent-name">{agent.name}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Metrics Input */}
        <div className="panel metrics-panel">
          <h2>Agent Metrics</h2>
          
          <div className="metrics-grid">
            <MetricSlider
              label="Success Rate"
              value={agentState.success_rate}
              min={0}
              max={1}
              step={0.01}
              format={(v) => `${(v * 100).toFixed(0)}%`}
              onChange={(v) => updateMetric('success_rate', v)}
              warning={agentState.success_rate < 0.7}
            />
            
            <MetricSlider
              label="Accuracy Score"
              value={agentState.accuracy_score}
              min={0}
              max={1}
              step={0.01}
              format={(v) => `${(v * 100).toFixed(0)}%`}
              onChange={(v) => updateMetric('accuracy_score', v)}
              warning={agentState.accuracy_score < 0.75}
            />
            
            <MetricSlider
              label="Efficiency Score"
              value={agentState.efficiency_score}
              min={0}
              max={1}
              step={0.01}
              format={(v) => `${(v * 100).toFixed(0)}%`}
              onChange={(v) => updateMetric('efficiency_score', v)}
              warning={agentState.efficiency_score < 0.6}
            />
            
            <MetricSlider
              label="CPU Usage"
              value={agentState.cpu_usage_percent}
              min={0}
              max={100}
              step={1}
              format={(v) => `${v}%`}
              onChange={(v) => updateMetric('cpu_usage_percent', v)}
              warning={agentState.cpu_usage_percent > 80}
            />
            
            <MetricSlider
              label="Response Latency"
              value={agentState.response_latency_ms}
              min={50}
              max={2000}
              step={10}
              format={(v) => `${v}ms`}
              onChange={(v) => updateMetric('response_latency_ms', v)}
              warning={agentState.response_latency_ms > 500}
            />
            
            <MetricSlider
              label="Memory Usage"
              value={agentState.memory_usage_mb}
              min={100}
              max={1000}
              step={10}
              format={(v) => `${v}MB`}
              onChange={(v) => updateMetric('memory_usage_mb', v)}
              warning={agentState.memory_usage_mb > 400}
            />
          </div>

          <button
            onClick={assessRisk}
            disabled={loading || !selectedAgent}
            className="assess-btn"
          >
            {loading ? '⏳ Assessing...' : '🔍 Assess Risk'}
          </button>
        </div>

        {/* Risk Gauge */}
        <div className="panel risk-gauge-panel">
          <h2>Risk Score</h2>
          
          {riskData ? (
            <div className="gauge-container">
              <svg viewBox="0 0 200 120" className="gauge-svg">
                {/* Background arc */}
                <path
                  d="M 20 100 A 80 80 0 0 1 180 100"
                  fill="none"
                  stroke="#e0e0e0"
                  strokeWidth="20"
                  strokeLinecap="round"
                />
                
                {/* Risk zones */}
                <path
                  d="M 20 100 A 80 80 0 0 1 65 35"
                  fill="none"
                  stroke="#4caf50"
                  strokeWidth="20"
                  strokeLinecap="round"
                />
                <path
                  d="M 65 35 A 80 80 0 0 1 135 35"
                  fill="none"
                  stroke="#ff9800"
                  strokeWidth="20"
                />
                <path
                  d="M 135 35 A 80 80 0 0 1 180 100"
                  fill="none"
                  stroke="#f44336"
                  strokeWidth="20"
                  strokeLinecap="round"
                />
                
                {/* Needle */}
                <line
                  x1="100"
                  y1="100"
                  x2="100"
                  y2="30"
                  stroke="#333"
                  strokeWidth="3"
                  strokeLinecap="round"
                  transform={`rotate(${getGaugeRotation(riskData.risk_score)}, 100, 100)`}
                />
                <circle cx="100" cy="100" r="8" fill="#333" />
              </svg>
              
              <div className="gauge-value">
                <span className="score">{riskData.risk_score.toFixed(1)}</span>
                <span 
                  className="level"
                  style={{ color: RISK_COLORS[riskData.risk_level] }}
                >
                  {riskData.risk_level}
                </span>
              </div>
              
              <div className="failure-probability">
                <span className="label">Failure Probability</span>
                <span className="value">{(riskData.failure_probability * 100).toFixed(1)}%</span>
              </div>
            </div>
          ) : (
            <div className="gauge-placeholder">
              Select an agent and assess risk to see results
            </div>
          )}
        </div>

        {/* Contributing Factors */}
        <div className="panel factors-panel">
          <h2>Contributing Factors</h2>
          
          {riskData?.contributing_factors ? (
            <div className="factors-list">
              {riskData.contributing_factors.map((factor, idx) => (
                <div 
                  key={idx}
                  className={`factor-item impact-${factor.impact}`}
                >
                  <div className="factor-header">
                    <span className="factor-name">{factor.factor}</span>
                    <span className={`impact-badge ${factor.impact}`}>
                      {factor.impact}
                    </span>
                  </div>
                  {factor.current_value !== null && (
                    <div className="factor-values">
                      <span>Current: {typeof factor.current_value === 'number' 
                        ? factor.current_value.toFixed(2) 
                        : factor.current_value}</span>
                      {factor.threshold && (
                        <span>Threshold: {factor.threshold}</span>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="factors-placeholder">
              No risk factors to display
            </div>
          )}
        </div>

        {/* Mitigation Steps */}
        <div className="panel mitigation-panel">
          <h2>Mitigation Steps</h2>
          
          {riskData?.mitigation_steps ? (
            <ol className="mitigation-list">
              {riskData.mitigation_steps.map((step, idx) => (
                <li key={idx} className="mitigation-item">
                  <span className="step-number">{idx + 1}</span>
                  <span className="step-text">{step}</span>
                </li>
              ))}
            </ol>
          ) : (
            <div className="mitigation-placeholder">
              Assess risk to see mitigation recommendations
            </div>
          )}
        </div>

        {/* Risk Heat Map */}
        <div className="panel heatmap-panel">
          <h2>Risk Heat Map</h2>
          <div className="heatmap">
            {[
              { label: 'Success', value: agentState.success_rate, invert: true },
              { label: 'Accuracy', value: agentState.accuracy_score, invert: true },
              { label: 'Efficiency', value: agentState.efficiency_score, invert: true },
              { label: 'CPU Load', value: agentState.cpu_usage_percent / 100, invert: false },
              { label: 'Latency', value: Math.min(agentState.response_latency_ms / 1000, 1), invert: false },
              { label: 'Memory', value: agentState.memory_usage_mb / 1000, invert: false }
            ].map((item, idx) => {
              const riskValue = item.invert ? 1 - item.value : item.value;
              const hue = 120 - (riskValue * 120); // Green to Red
              return (
                <div 
                  key={idx}
                  className="heatmap-cell"
                  style={{ backgroundColor: `hsl(${hue}, 70%, 50%)` }}
                >
                  <span className="cell-label">{item.label}</span>
                  <span className="cell-value">
                    {item.invert 
                      ? `${(item.value * 100).toFixed(0)}%`
                      : `${(item.value * 100).toFixed(0)}%`
                    }
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="error-toast">
          ⚠️ {error}
          <button onClick={() => setError(null)}>×</button>
        </div>
      )}

      <style jsx>{`
        .risk-dashboard {
          min-height: 100vh;
          background: #1a1a2e;
          color: white;
          padding: 1.5rem;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        .dashboard-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1.5rem;
        }

        .dashboard-header h1 {
          margin: 0;
          font-size: 1.75rem;
        }

        .auto-refresh {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          color: #aaa;
          cursor: pointer;
        }

        .dashboard-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 1rem;
        }

        .panel {
          background: #16213e;
          border-radius: 12px;
          padding: 1.25rem;
        }

        .panel h2 {
          margin: 0 0 1rem;
          font-size: 1rem;
          color: #aaa;
          text-transform: uppercase;
          letter-spacing: 1px;
        }

        .agent-list {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .agent-btn {
          display: flex;
          flex-direction: column;
          align-items: flex-start;
          padding: 0.75rem 1rem;
          background: #0f3460;
          border: 2px solid transparent;
          border-radius: 8px;
          color: white;
          cursor: pointer;
          transition: all 0.2s;
        }

        .agent-btn:hover {
          background: #1a4a7a;
        }

        .agent-btn.active {
          border-color: #4caf50;
          background: #1a4a7a;
        }

        .agent-id {
          font-size: 0.75rem;
          color: #888;
        }

        .agent-name {
          font-weight: 500;
        }

        .metrics-grid {
          display: grid;
          gap: 1rem;
          margin-bottom: 1rem;
        }

        .assess-btn {
          width: 100%;
          padding: 0.75rem;
          background: #e94560;
          border: none;
          border-radius: 8px;
          color: white;
          font-weight: 600;
          cursor: pointer;
          transition: background 0.2s;
        }

        .assess-btn:hover:not(:disabled) {
          background: #ff6b6b;
        }

        .assess-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .gauge-container {
          text-align: center;
        }

        .gauge-svg {
          width: 100%;
          max-width: 200px;
        }

        .gauge-value {
          margin-top: -1rem;
        }

        .gauge-value .score {
          display: block;
          font-size: 3rem;
          font-weight: bold;
        }

        .gauge-value .level {
          font-size: 1.25rem;
          font-weight: 600;
        }

        .failure-probability {
          margin-top: 1rem;
          padding: 0.75rem;
          background: rgba(255,255,255,0.1);
          border-radius: 8px;
        }

        .failure-probability .label {
          display: block;
          font-size: 0.75rem;
          color: #888;
        }

        .failure-probability .value {
          font-size: 1.5rem;
          font-weight: bold;
        }

        .gauge-placeholder,
        .factors-placeholder,
        .mitigation-placeholder {
          color: #666;
          text-align: center;
          padding: 2rem;
        }

        .factors-list {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }

        .factor-item {
          padding: 0.75rem;
          background: rgba(255,255,255,0.05);
          border-radius: 8px;
          border-left: 3px solid #666;
        }

        .factor-item.impact-high {
          border-left-color: #f44336;
        }

        .factor-item.impact-medium {
          border-left-color: #ff9800;
        }

        .factor-item.impact-low {
          border-left-color: #4caf50;
        }

        .factor-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .factor-name {
          font-weight: 500;
        }

        .impact-badge {
          font-size: 0.7rem;
          padding: 0.2rem 0.5rem;
          border-radius: 4px;
          text-transform: uppercase;
        }

        .impact-badge.high {
          background: #f44336;
        }

        .impact-badge.medium {
          background: #ff9800;
        }

        .impact-badge.low {
          background: #4caf50;
        }

        .factor-values {
          display: flex;
          gap: 1rem;
          margin-top: 0.5rem;
          font-size: 0.8rem;
          color: #888;
        }

        .mitigation-list {
          list-style: none;
          padding: 0;
          margin: 0;
        }

        .mitigation-item {
          display: flex;
          gap: 0.75rem;
          padding: 0.75rem 0;
          border-bottom: 1px solid rgba(255,255,255,0.1);
        }

        .step-number {
          width: 24px;
          height: 24px;
          background: #e94560;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 0.75rem;
          font-weight: bold;
          flex-shrink: 0;
        }

        .step-text {
          line-height: 1.4;
        }

        .heatmap {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 0.5rem;
        }

        .heatmap-cell {
          aspect-ratio: 1;
          border-radius: 8px;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          transition: transform 0.2s;
        }

        .heatmap-cell:hover {
          transform: scale(1.05);
        }

        .cell-label {
          font-size: 0.7rem;
          opacity: 0.9;
        }

        .cell-value {
          font-size: 1.25rem;
          font-weight: bold;
        }

        .error-toast {
          position: fixed;
          bottom: 1rem;
          right: 1rem;
          background: #f44336;
          padding: 1rem 1.5rem;
          border-radius: 8px;
          display: flex;
          align-items: center;
          gap: 1rem;
        }

        .error-toast button {
          background: none;
          border: none;
          color: white;
          font-size: 1.25rem;
          cursor: pointer;
        }

        @media (max-width: 1024px) {
          .dashboard-grid {
            grid-template-columns: repeat(2, 1fr);
          }
        }

        @media (max-width: 768px) {
          .dashboard-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
};

// Metric Slider Component
const MetricSlider = ({ label, value, min, max, step, format, onChange, warning }) => (
  <div className={`metric-slider ${warning ? 'warning' : ''}`}>
    <div className="metric-header">
      <span className="metric-label">{label}</span>
      <span className="metric-value">{format(value)}</span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
    />
    <style jsx>{`
      .metric-slider {
        padding: 0.5rem;
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        border-left: 3px solid #4caf50;
      }

      .metric-slider.warning {
        border-left-color: #ff9800;
      }

      .metric-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
      }

      .metric-label {
        font-size: 0.8rem;
        color: #aaa;
      }

      .metric-value {
        font-weight: 600;
      }

      input[type="range"] {
        width: 100%;
        height: 4px;
        -webkit-appearance: none;
        background: #333;
        border-radius: 2px;
      }

      input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        width: 16px;
        height: 16px;
        background: #e94560;
        border-radius: 50%;
        cursor: pointer;
      }
    `}</style>
  </div>
);

export default RiskAssessmentDashboard;

