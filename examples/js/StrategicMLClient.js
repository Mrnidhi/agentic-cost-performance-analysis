/**
 * AI Agent Performance Intelligence System - JavaScript API Client
 * Course: DATA 230 (Data Visualization) at SJSU
 * 
 * Usage:
 *   const client = new StrategicMLClient('http://localhost:8000');
 *   const config = await client.getOptimalConfiguration({ task_complexity: 7 });
 */

class StrategicMLClient {
  constructor(baseUrl = 'http://localhost:8000', options = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.timeout = options.timeout || 30000;
    this.headers = {
      'Content-Type': 'application/json',
      ...options.headers
    };
    this.retryAttempts = options.retryAttempts || 3;
    this.retryDelay = options.retryDelay || 1000;
  }

  /**
   * Make HTTP request with retry logic
   */
  async _request(endpoint, method = 'GET', body = null) {
    const url = `${this.baseUrl}${endpoint}`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    let lastError;
    
    for (let attempt = 1; attempt <= this.retryAttempts; attempt++) {
      try {
        const options = {
          method,
          headers: this.headers,
          signal: controller.signal
        };

        if (body && method !== 'GET') {
          options.body = JSON.stringify(body);
        }

        const response = await fetch(url, options);
        clearTimeout(timeoutId);

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new APIError(
            errorData.detail || `HTTP ${response.status}: ${response.statusText}`,
            response.status,
            errorData
          );
        }

        return await response.json();
      } catch (error) {
        lastError = error;
        
        if (error.name === 'AbortError') {
          throw new APIError('Request timeout', 408);
        }
        
        if (attempt < this.retryAttempts && this._shouldRetry(error)) {
          await this._delay(this.retryDelay * attempt);
          continue;
        }
        
        throw error instanceof APIError ? error : new APIError(error.message, 0);
      }
    }
    
    throw lastError;
  }

  _shouldRetry(error) {
    if (error instanceof APIError) {
      return error.status >= 500 || error.status === 429;
    }
    return true;
  }

  _delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // ===========================================================================
  // Health Check
  // ===========================================================================

  /**
   * Check API health status
   * @returns {Promise<Object>} Health status
   */
  async healthCheck() {
    return this._request('/health');
  }

  // ===========================================================================
  // Optimization Endpoints
  // ===========================================================================

  /**
   * Get optimal agent configuration for task requirements
   * @param {Object} taskRequirements - Task requirements
   * @param {number} taskRequirements.task_complexity - Task complexity (1-10)
   * @param {number} taskRequirements.autonomy_level - Required autonomy level (1-10)
   * @param {number} [taskRequirements.min_success_rate=0.8] - Minimum success rate
   * @param {number} [taskRequirements.min_accuracy=0.8] - Minimum accuracy
   * @param {number} [taskRequirements.max_cost_cents=0.02] - Maximum cost per task
   * @param {number} [taskRequirements.max_latency_ms=500] - Maximum latency
   * @param {string} [taskRequirements.deployment_environment] - Preferred environment
   * @param {string} [taskRequirements.task_category] - Task category
   * @returns {Promise<Object>} Optimal configuration with trade-off analysis
   */
  async getOptimalConfiguration(taskRequirements) {
    const payload = {
      task_complexity: taskRequirements.task_complexity || 5,
      autonomy_level: taskRequirements.autonomy_level || 5,
      min_success_rate: taskRequirements.min_success_rate || 0.8,
      min_accuracy: taskRequirements.min_accuracy || 0.8,
      max_cost_cents: taskRequirements.max_cost_cents || 0.02,
      max_latency_ms: taskRequirements.max_latency_ms || 500,
      deployment_environment: taskRequirements.deployment_environment || null,
      task_category: taskRequirements.task_category || null
    };

    return this._request('/v1/optimize-agent-configuration', 'POST', payload);
  }

  // ===========================================================================
  // Benchmarking Endpoints
  // ===========================================================================

  /**
   * Get performance benchmark for agent
   * @param {Object} agentData - Agent metrics data
   * @param {string} agentData.agent_type - Type of AI agent
   * @param {string} agentData.model_architecture - Model architecture
   * @param {number} agentData.success_rate - Current success rate (0-1)
   * @param {number} agentData.accuracy_score - Current accuracy score (0-1)
   * @param {number} agentData.efficiency_score - Current efficiency score (0-1)
   * @param {number} agentData.cost_per_task_cents - Cost per task in cents
   * @param {number} agentData.execution_time_seconds - Execution time
   * @param {number} agentData.response_latency_ms - Response latency
   * @param {number} agentData.memory_usage_mb - Memory usage
   * @param {number} agentData.cpu_usage_percent - CPU usage
   * @param {number} [agentData.error_recovery_rate=0.5] - Error recovery rate
   * @returns {Promise<Object>} Benchmark results with recommendations
   */
  async getPerformanceBenchmark(agentData) {
    const payload = {
      agent_type: agentData.agent_type,
      model_architecture: agentData.model_architecture,
      success_rate: agentData.success_rate,
      accuracy_score: agentData.accuracy_score,
      efficiency_score: agentData.efficiency_score,
      cost_per_task_cents: agentData.cost_per_task_cents,
      execution_time_seconds: agentData.execution_time_seconds,
      response_latency_ms: agentData.response_latency_ms,
      memory_usage_mb: agentData.memory_usage_mb,
      cpu_usage_percent: agentData.cpu_usage_percent,
      error_recovery_rate: agentData.error_recovery_rate || 0.5
    };

    return this._request('/v1/performance-benchmarking', 'POST', payload);
  }

  // ===========================================================================
  // Cost-Performance Tradeoffs
  // ===========================================================================

  /**
   * Get cost-performance tradeoff analysis
   * @param {Object} constraints - Budget constraints
   * @param {number} constraints.max_budget_cents - Maximum budget per task
   * @param {number} [constraints.min_performance_index=0.5] - Minimum performance
   * @param {string} [constraints.risk_tolerance='medium'] - Risk tolerance: low, medium, high
   * @param {string} [constraints.optimization_priority='balanced'] - Priority: cost, performance, balanced
   * @returns {Promise<Object>} Pareto-optimal options with business impact
   */
  async getCostTradeoffs(constraints) {
    const payload = {
      max_budget_cents: constraints.max_budget_cents,
      min_performance_index: constraints.min_performance_index || 0.5,
      risk_tolerance: constraints.risk_tolerance || 'medium',
      optimization_priority: constraints.optimization_priority || 'balanced'
    };

    return this._request('/v1/cost-performance-tradeoffs', 'POST', payload);
  }

  // ===========================================================================
  // Risk Assessment
  // ===========================================================================

  /**
   * Get failure risk assessment for agent
   * @param {Object} agentState - Current agent state
   * @param {string} agentState.agent_id - Agent identifier
   * @param {number} agentState.success_rate - Current success rate (0-1)
   * @param {number} agentState.accuracy_score - Current accuracy score (0-1)
   * @param {number} agentState.efficiency_score - Current efficiency score (0-1)
   * @param {number} agentState.execution_time_seconds - Execution time
   * @param {number} agentState.response_latency_ms - Response latency
   * @param {number} agentState.memory_usage_mb - Memory usage
   * @param {number} agentState.cpu_usage_percent - CPU usage
   * @param {number} [agentState.error_recovery_rate=0.5] - Error recovery rate
   * @returns {Promise<Object>} Risk score, failure probability, mitigation steps
   */
  async getFailureRisk(agentState) {
    const payload = {
      agent_id: agentState.agent_id,
      success_rate: agentState.success_rate,
      accuracy_score: agentState.accuracy_score,
      efficiency_score: agentState.efficiency_score,
      execution_time_seconds: agentState.execution_time_seconds,
      response_latency_ms: agentState.response_latency_ms,
      memory_usage_mb: agentState.memory_usage_mb,
      cpu_usage_percent: agentState.cpu_usage_percent,
      error_recovery_rate: agentState.error_recovery_rate || 0.5
    };

    return this._request('/v1/failure-risk-assessment', 'POST', payload);
  }

  // ===========================================================================
  // Agent Recommendations
  // ===========================================================================

  /**
   * Get agent recommendations for task profile
   * @param {Object} taskProfile - Task profile
   * @param {number} taskProfile.task_complexity - Task complexity (1-10)
   * @param {number} taskProfile.autonomy_level - Required autonomy level (1-10)
   * @param {number} [taskProfile.min_success_rate=0.8] - Minimum success rate
   * @param {number} [taskProfile.min_accuracy=0.8] - Minimum accuracy
   * @param {number} [taskProfile.min_efficiency=0.7] - Minimum efficiency
   * @param {number} [taskProfile.max_cost_cents=0.02] - Maximum cost
   * @param {number} [taskProfile.min_performance_index=0.6] - Minimum performance
   * @param {number} [taskProfile.min_cost_efficiency=50] - Minimum cost efficiency
   * @param {number} [taskProfile.top_k=5] - Number of recommendations
   * @returns {Promise<Object>} Ranked agent recommendations
   */
  async getAgentRecommendations(taskProfile) {
    const payload = {
      task_complexity: taskProfile.task_complexity || 5,
      autonomy_level: taskProfile.autonomy_level || 5,
      min_success_rate: taskProfile.min_success_rate || 0.8,
      min_accuracy: taskProfile.min_accuracy || 0.8,
      min_efficiency: taskProfile.min_efficiency || 0.7,
      max_cost_cents: taskProfile.max_cost_cents || 0.02,
      min_performance_index: taskProfile.min_performance_index || 0.6,
      min_cost_efficiency: taskProfile.min_cost_efficiency || 50,
      top_k: taskProfile.top_k || 5
    };

    return this._request('/v1/agent-recommendation-engine', 'POST', payload);
  }
}

/**
 * Custom API Error class
 */
class APIError extends Error {
  constructor(message, status, data = null) {
    super(message);
    this.name = 'APIError';
    this.status = status;
    this.data = data;
  }
}

// Export for different module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { StrategicMLClient, APIError };
}

if (typeof window !== 'undefined') {
  window.StrategicMLClient = StrategicMLClient;
  window.APIError = APIError;
}

export { StrategicMLClient, APIError };

