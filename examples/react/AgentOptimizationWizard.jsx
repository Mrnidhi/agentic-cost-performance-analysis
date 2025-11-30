/**
 * Agent Optimization Wizard Component
 * Course: DATA 230 (Data Visualization) at SJSU
 * 
 * A multi-step wizard for configuring and optimizing AI agent deployments.
 */

import React, { useState, useCallback } from 'react';
import { StrategicMLClient } from '../js/StrategicMLClient';

// Initialize API client
const apiClient = new StrategicMLClient(process.env.REACT_APP_API_URL || 'http://localhost:8000');

// Wizard steps
const STEPS = {
  REQUIREMENTS: 0,
  CONSTRAINTS: 1,
  OPTIMIZATION: 2,
  RESULTS: 3
};

const AgentOptimizationWizard = () => {
  // Form state
  const [currentStep, setCurrentStep] = useState(STEPS.REQUIREMENTS);
  const [formData, setFormData] = useState({
    task_complexity: 5,
    autonomy_level: 5,
    min_success_rate: 0.8,
    min_accuracy: 0.8,
    max_cost_cents: 0.02,
    max_latency_ms: 500,
    deployment_environment: 'Cloud',
    task_category: 'Code Generation'
  });

  // API state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);

  // Handle form changes
  const handleChange = useCallback((field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  }, []);

  // Submit optimization request
  const handleOptimize = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await apiClient.getOptimalConfiguration(formData);
      setResults(response);
      setCurrentStep(STEPS.RESULTS);
    } catch (err) {
      setError(err.message || 'Failed to get optimization results');
    } finally {
      setLoading(false);
    }
  }, [formData]);

  // Navigation
  const nextStep = () => setCurrentStep(prev => Math.min(prev + 1, STEPS.RESULTS));
  const prevStep = () => setCurrentStep(prev => Math.max(prev - 1, STEPS.REQUIREMENTS));
  const resetWizard = () => {
    setCurrentStep(STEPS.REQUIREMENTS);
    setResults(null);
    setError(null);
  };

  return (
    <div className="wizard-container">
      <h1>Agent Optimization Wizard</h1>
      
      {/* Progress indicator */}
      <div className="progress-bar">
        {['Requirements', 'Constraints', 'Optimize', 'Results'].map((label, idx) => (
          <div 
            key={idx}
            className={`progress-step ${currentStep >= idx ? 'active' : ''}`}
          >
            <span className="step-number">{idx + 1}</span>
            <span className="step-label">{label}</span>
          </div>
        ))}
      </div>

      {/* Error display */}
      {error && (
        <div className="error-banner">
          <span>⚠️ {error}</span>
          <button onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}

      {/* Step 1: Task Requirements */}
      {currentStep === STEPS.REQUIREMENTS && (
        <div className="wizard-step">
          <h2>Task Requirements</h2>
          
          <div className="form-group">
            <label>Task Complexity (1-10)</label>
            <input
              type="range"
              min="1"
              max="10"
              value={formData.task_complexity}
              onChange={(e) => handleChange('task_complexity', parseInt(e.target.value))}
            />
            <span className="value-display">{formData.task_complexity}</span>
          </div>

          <div className="form-group">
            <label>Autonomy Level (1-10)</label>
            <input
              type="range"
              min="1"
              max="10"
              value={formData.autonomy_level}
              onChange={(e) => handleChange('autonomy_level', parseInt(e.target.value))}
            />
            <span className="value-display">{formData.autonomy_level}</span>
          </div>

          <div className="form-group">
            <label>Task Category</label>
            <select
              value={formData.task_category}
              onChange={(e) => handleChange('task_category', e.target.value)}
            >
              <option value="Code Generation">Code Generation</option>
              <option value="Data Analysis">Data Analysis</option>
              <option value="Content Writing">Content Writing</option>
              <option value="Research">Research</option>
              <option value="Decision Making">Decision Making</option>
            </select>
          </div>

          <div className="form-group">
            <label>Deployment Environment</label>
            <select
              value={formData.deployment_environment}
              onChange={(e) => handleChange('deployment_environment', e.target.value)}
            >
              <option value="Cloud">Cloud</option>
              <option value="Edge">Edge</option>
              <option value="Hybrid">Hybrid</option>
              <option value="Server">Server</option>
            </select>
          </div>

          <div className="button-group">
            <button onClick={nextStep} className="btn-primary">
              Next: Set Constraints →
            </button>
          </div>
        </div>
      )}

      {/* Step 2: Constraints */}
      {currentStep === STEPS.CONSTRAINTS && (
        <div className="wizard-step">
          <h2>Performance Constraints</h2>

          <div className="form-group">
            <label>Minimum Success Rate</label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.05"
              value={formData.min_success_rate}
              onChange={(e) => handleChange('min_success_rate', parseFloat(e.target.value))}
            />
          </div>

          <div className="form-group">
            <label>Minimum Accuracy</label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.05"
              value={formData.min_accuracy}
              onChange={(e) => handleChange('min_accuracy', parseFloat(e.target.value))}
            />
          </div>

          <div className="form-group">
            <label>Maximum Cost (cents/task)</label>
            <input
              type="number"
              min="0.001"
              max="0.1"
              step="0.001"
              value={formData.max_cost_cents}
              onChange={(e) => handleChange('max_cost_cents', parseFloat(e.target.value))}
            />
          </div>

          <div className="form-group">
            <label>Maximum Latency (ms)</label>
            <input
              type="number"
              min="50"
              max="5000"
              step="50"
              value={formData.max_latency_ms}
              onChange={(e) => handleChange('max_latency_ms', parseInt(e.target.value))}
            />
          </div>

          <div className="button-group">
            <button onClick={prevStep} className="btn-secondary">
              ← Back
            </button>
            <button onClick={nextStep} className="btn-primary">
              Next: Optimize →
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Optimization */}
      {currentStep === STEPS.OPTIMIZATION && (
        <div className="wizard-step">
          <h2>Optimization Summary</h2>

          <div className="summary-card">
            <h3>Configuration</h3>
            <ul>
              <li><strong>Task Complexity:</strong> {formData.task_complexity}/10</li>
              <li><strong>Autonomy Level:</strong> {formData.autonomy_level}/10</li>
              <li><strong>Category:</strong> {formData.task_category}</li>
              <li><strong>Environment:</strong> {formData.deployment_environment}</li>
            </ul>

            <h3>Constraints</h3>
            <ul>
              <li><strong>Min Success Rate:</strong> {(formData.min_success_rate * 100).toFixed(0)}%</li>
              <li><strong>Min Accuracy:</strong> {(formData.min_accuracy * 100).toFixed(0)}%</li>
              <li><strong>Max Cost:</strong> ${formData.max_cost_cents.toFixed(4)}/task</li>
              <li><strong>Max Latency:</strong> {formData.max_latency_ms}ms</li>
            </ul>
          </div>

          <div className="button-group">
            <button onClick={prevStep} className="btn-secondary" disabled={loading}>
              ← Back
            </button>
            <button 
              onClick={handleOptimize} 
              className="btn-primary"
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="spinner"></span>
                  Optimizing...
                </>
              ) : (
                '🚀 Run Optimization'
              )}
            </button>
          </div>
        </div>
      )}

      {/* Step 4: Results */}
      {currentStep === STEPS.RESULTS && results && (
        <div className="wizard-step">
          <h2>Optimization Results</h2>

          <div className="results-grid">
            {/* Recommended Configuration */}
            <div className="result-card primary">
              <h3>✅ Recommended Configuration</h3>
              <div className="config-details">
                <p><strong>Deployment:</strong> {results.optimal_configuration.deployment_environment}</p>
                <p><strong>Memory:</strong> {results.optimal_configuration.recommended_memory_mb} MB</p>
                <p><strong>CPU:</strong> {results.optimal_configuration.recommended_cpu_percent}%</p>
                <p><strong>Est. Cost:</strong> ${results.optimal_configuration.estimated_cost_cents.toFixed(4)}/task</p>
                <p><strong>Est. Success:</strong> {(results.optimal_configuration.estimated_success_rate * 100).toFixed(1)}%</p>
              </div>
            </div>

            {/* Trade-off Analysis */}
            <div className="result-card">
              <h3>⚖️ Trade-off Analysis</h3>
              <ul>
                {Object.entries(results.tradeoff_analysis).map(([key, value]) => (
                  <li key={key}>
                    <strong>{key.replace(/_/g, ' ')}:</strong> {value}
                  </li>
                ))}
              </ul>
            </div>

            {/* Confidence Score */}
            <div className="result-card">
              <h3>📊 Confidence</h3>
              <div className="confidence-meter">
                <div 
                  className="confidence-fill"
                  style={{ width: `${results.confidence_score * 100}%` }}
                />
                <span>{(results.confidence_score * 100).toFixed(0)}%</span>
              </div>
            </div>

            {/* Recommendations */}
            <div className="result-card full-width">
              <h3>💡 Recommendations</h3>
              <ul className="recommendations-list">
                {results.recommendations.map((rec, idx) => (
                  <li key={idx}>{rec}</li>
                ))}
              </ul>
            </div>
          </div>

          <div className="button-group">
            <button onClick={resetWizard} className="btn-secondary">
              ↺ Start Over
            </button>
            <button 
              onClick={() => navigator.clipboard.writeText(JSON.stringify(results, null, 2))}
              className="btn-primary"
            >
              📋 Copy Results
            </button>
          </div>
        </div>
      )}

      {/* Styles */}
      <style jsx>{`
        .wizard-container {
          max-width: 800px;
          margin: 0 auto;
          padding: 2rem;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        .progress-bar {
          display: flex;
          justify-content: space-between;
          margin-bottom: 2rem;
        }

        .progress-step {
          display: flex;
          flex-direction: column;
          align-items: center;
          opacity: 0.5;
        }

        .progress-step.active {
          opacity: 1;
        }

        .step-number {
          width: 32px;
          height: 32px;
          border-radius: 50%;
          background: #e0e0e0;
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: bold;
        }

        .progress-step.active .step-number {
          background: #2196f3;
          color: white;
        }

        .wizard-step {
          background: #f5f5f5;
          padding: 2rem;
          border-radius: 8px;
        }

        .form-group {
          margin-bottom: 1.5rem;
        }

        .form-group label {
          display: block;
          margin-bottom: 0.5rem;
          font-weight: 500;
        }

        .form-group input,
        .form-group select {
          width: 100%;
          padding: 0.75rem;
          border: 1px solid #ddd;
          border-radius: 4px;
          font-size: 1rem;
        }

        .value-display {
          margin-left: 1rem;
          font-weight: bold;
          color: #2196f3;
        }

        .button-group {
          display: flex;
          gap: 1rem;
          margin-top: 2rem;
        }

        .btn-primary,
        .btn-secondary {
          padding: 0.75rem 1.5rem;
          border: none;
          border-radius: 4px;
          font-size: 1rem;
          cursor: pointer;
          transition: all 0.2s;
        }

        .btn-primary {
          background: #2196f3;
          color: white;
        }

        .btn-primary:hover {
          background: #1976d2;
        }

        .btn-secondary {
          background: #e0e0e0;
          color: #333;
        }

        .btn-primary:disabled {
          opacity: 0.7;
          cursor: not-allowed;
        }

        .error-banner {
          background: #ffebee;
          color: #c62828;
          padding: 1rem;
          border-radius: 4px;
          margin-bottom: 1rem;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .spinner {
          display: inline-block;
          width: 16px;
          height: 16px;
          border: 2px solid #fff;
          border-top-color: transparent;
          border-radius: 50%;
          animation: spin 1s linear infinite;
          margin-right: 0.5rem;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .results-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 1rem;
        }

        .result-card {
          background: white;
          padding: 1.5rem;
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .result-card.primary {
          border-left: 4px solid #4caf50;
        }

        .result-card.full-width {
          grid-column: 1 / -1;
        }

        .confidence-meter {
          height: 24px;
          background: #e0e0e0;
          border-radius: 12px;
          overflow: hidden;
          position: relative;
        }

        .confidence-fill {
          height: 100%;
          background: linear-gradient(90deg, #4caf50, #8bc34a);
          transition: width 0.5s ease;
        }

        .confidence-meter span {
          position: absolute;
          right: 10px;
          top: 50%;
          transform: translateY(-50%);
          font-weight: bold;
        }

        .recommendations-list li {
          padding: 0.5rem 0;
          border-bottom: 1px solid #eee;
        }

        .summary-card {
          background: white;
          padding: 1.5rem;
          border-radius: 8px;
          margin-bottom: 1rem;
        }

        .summary-card h3 {
          margin-top: 1rem;
          color: #666;
        }

        .summary-card ul {
          list-style: none;
          padding: 0;
        }

        .summary-card li {
          padding: 0.25rem 0;
        }
      `}</style>
    </div>
  );
};

export default AgentOptimizationWizard;

