/**
 * Performance Tradeoff Visualization Component
 * Course: DATA 230 (Data Visualization) at SJSU
 * 
 * Interactive Pareto frontier visualization for cost-performance tradeoffs.
 */

import React, { useState, useEffect, useMemo } from 'react';
import { StrategicMLClient } from '../js/StrategicMLClient';

const apiClient = new StrategicMLClient(process.env.REACT_APP_API_URL || 'http://localhost:8000');

const PerformanceTradeoffVisualization = () => {
  // State
  const [constraints, setConstraints] = useState({
    max_budget_cents: 0.03,
    min_performance_index: 0.5,
    risk_tolerance: 'medium',
    optimization_priority: 'balanced'
  });
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedOption, setSelectedOption] = useState(null);

  // Fetch tradeoff data
  const fetchTradeoffs = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await apiClient.getCostTradeoffs(constraints);
      setData(response);
      setSelectedOption(response.recommended_option);
    } catch (err) {
      setError(err.message || 'Failed to fetch tradeoff data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTradeoffs();
  }, []);

  // Calculate chart dimensions
  const chartConfig = useMemo(() => {
    if (!data?.pareto_options) return null;

    const costs = data.pareto_options.map(o => o.estimated_cost_cents || o.cost);
    const performances = data.pareto_options.map(o => o.estimated_performance || o.performance);

    return {
      minCost: Math.min(...costs) * 0.8,
      maxCost: Math.max(...costs) * 1.2,
      minPerf: Math.min(...performances) * 0.9,
      maxPerf: Math.max(...performances) * 1.05,
      width: 600,
      height: 400,
      padding: 60
    };
  }, [data]);

  // Convert data point to SVG coordinates
  const toSvgCoords = (cost, performance) => {
    if (!chartConfig) return { x: 0, y: 0 };
    
    const { minCost, maxCost, minPerf, maxPerf, width, height, padding } = chartConfig;
    
    const x = padding + ((cost - minCost) / (maxCost - minCost)) * (width - 2 * padding);
    const y = height - padding - ((performance - minPerf) / (maxPerf - minPerf)) * (height - 2 * padding);
    
    return { x, y };
  };

  // Generate Pareto frontier path
  const paretoPath = useMemo(() => {
    if (!data?.pareto_options || !chartConfig) return '';

    const sortedOptions = [...data.pareto_options].sort(
      (a, b) => (a.estimated_cost_cents || a.cost) - (b.estimated_cost_cents || b.cost)
    );

    const points = sortedOptions.map(opt => {
      const { x, y } = toSvgCoords(
        opt.estimated_cost_cents || opt.cost,
        opt.estimated_performance || opt.performance
      );
      return `${x},${y}`;
    });

    return `M ${points.join(' L ')}`;
  }, [data, chartConfig]);

  return (
    <div className="tradeoff-container">
      <h1>Cost-Performance Tradeoff Analysis</h1>

      {/* Controls */}
      <div className="controls-panel">
        <div className="control-group">
          <label>Max Budget (cents/task)</label>
          <input
            type="range"
            min="0.005"
            max="0.05"
            step="0.005"
            value={constraints.max_budget_cents}
            onChange={(e) => setConstraints(prev => ({
              ...prev,
              max_budget_cents: parseFloat(e.target.value)
            }))}
          />
          <span>${constraints.max_budget_cents.toFixed(3)}</span>
        </div>

        <div className="control-group">
          <label>Min Performance</label>
          <input
            type="range"
            min="0.3"
            max="0.9"
            step="0.05"
            value={constraints.min_performance_index}
            onChange={(e) => setConstraints(prev => ({
              ...prev,
              min_performance_index: parseFloat(e.target.value)
            }))}
          />
          <span>{(constraints.min_performance_index * 100).toFixed(0)}%</span>
        </div>

        <div className="control-group">
          <label>Optimization Priority</label>
          <select
            value={constraints.optimization_priority}
            onChange={(e) => setConstraints(prev => ({
              ...prev,
              optimization_priority: e.target.value
            }))}
          >
            <option value="cost">Cost Focused</option>
            <option value="balanced">Balanced</option>
            <option value="performance">Performance Focused</option>
          </select>
        </div>

        <button 
          onClick={fetchTradeoffs} 
          disabled={loading}
          className="analyze-btn"
        >
          {loading ? 'Analyzing...' : '🔄 Update Analysis'}
        </button>
      </div>

      {/* Error display */}
      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}

      {/* Chart */}
      {chartConfig && data && (
        <div className="chart-container">
          <svg width={chartConfig.width} height={chartConfig.height}>
            {/* Grid lines */}
            <defs>
              <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#eee" strokeWidth="1"/>
              </pattern>
            </defs>
            <rect 
              x={chartConfig.padding} 
              y={chartConfig.padding} 
              width={chartConfig.width - 2 * chartConfig.padding} 
              height={chartConfig.height - 2 * chartConfig.padding} 
              fill="url(#grid)"
            />

            {/* Axes */}
            <line
              x1={chartConfig.padding}
              y1={chartConfig.height - chartConfig.padding}
              x2={chartConfig.width - chartConfig.padding}
              y2={chartConfig.height - chartConfig.padding}
              stroke="#333"
              strokeWidth="2"
            />
            <line
              x1={chartConfig.padding}
              y1={chartConfig.padding}
              x2={chartConfig.padding}
              y2={chartConfig.height - chartConfig.padding}
              stroke="#333"
              strokeWidth="2"
            />

            {/* Axis labels */}
            <text
              x={chartConfig.width / 2}
              y={chartConfig.height - 15}
              textAnchor="middle"
              className="axis-label"
            >
              Cost (cents/task)
            </text>
            <text
              x={20}
              y={chartConfig.height / 2}
              textAnchor="middle"
              transform={`rotate(-90, 20, ${chartConfig.height / 2})`}
              className="axis-label"
            >
              Performance Index
            </text>

            {/* Pareto frontier line */}
            <path
              d={paretoPath}
              fill="none"
              stroke="#2196f3"
              strokeWidth="3"
              strokeDasharray="5,5"
            />

            {/* Feasible region */}
            <rect
              x={chartConfig.padding}
              y={chartConfig.padding}
              width={(constraints.max_budget_cents - chartConfig.minCost) / (chartConfig.maxCost - chartConfig.minCost) * (chartConfig.width - 2 * chartConfig.padding)}
              height={chartConfig.height - 2 * chartConfig.padding}
              fill="#e3f2fd"
              opacity="0.3"
            />

            {/* Data points */}
            {data.pareto_options.map((option, idx) => {
              const cost = option.estimated_cost_cents || option.cost;
              const perf = option.estimated_performance || option.performance;
              const { x, y } = toSvgCoords(cost, perf);
              const isSelected = selectedOption?.option_name === option.option_name;
              const isRecommended = data.recommended_option?.option_name === option.option_name;

              return (
                <g key={idx}>
                  {/* Point */}
                  <circle
                    cx={x}
                    cy={y}
                    r={isSelected ? 14 : 10}
                    fill={isRecommended ? '#4caf50' : '#2196f3'}
                    stroke={isSelected ? '#333' : 'white'}
                    strokeWidth={isSelected ? 3 : 2}
                    style={{ cursor: 'pointer' }}
                    onClick={() => setSelectedOption(option)}
                  />
                  
                  {/* Label */}
                  <text
                    x={x}
                    y={y - 18}
                    textAnchor="middle"
                    className="point-label"
                  >
                    {option.option_name}
                  </text>

                  {/* Recommended badge */}
                  {isRecommended && (
                    <text x={x + 16} y={y - 8} className="badge">★</text>
                  )}
                </g>
              );
            })}
          </svg>

          {/* Legend */}
          <div className="chart-legend">
            <div className="legend-item">
              <span className="dot recommended"></span>
              Recommended
            </div>
            <div className="legend-item">
              <span className="dot option"></span>
              Alternative
            </div>
            <div className="legend-item">
              <span className="line pareto"></span>
              Pareto Frontier
            </div>
          </div>
        </div>
      )}

      {/* Selected option details */}
      {selectedOption && (
        <div className="option-details">
          <h2>{selectedOption.option_name}</h2>
          
          <div className="details-grid">
            <div className="detail-card">
              <h3>💰 Cost</h3>
              <p className="value">
                ${(selectedOption.estimated_cost_cents || selectedOption.cost).toFixed(4)}
              </p>
              <p className="label">per task</p>
            </div>

            <div className="detail-card">
              <h3>📊 Performance</h3>
              <p className="value">
                {((selectedOption.estimated_performance || selectedOption.performance) * 100).toFixed(1)}%
              </p>
              <p className="label">index</p>
            </div>

            <div className="detail-card">
              <h3>⚠️ Risk</h3>
              <p className="value">{selectedOption.risk_level || 'Medium'}</p>
              <p className="label">level</p>
            </div>

            {selectedOption.recommended_architecture && (
              <div className="detail-card">
                <h3>🤖 Architecture</h3>
                <p className="value">{selectedOption.recommended_architecture}</p>
                <p className="label">recommended</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Business Impact */}
      {data?.business_impact && (
        <div className="business-impact">
          <h2>📈 Business Impact</h2>
          <div className="impact-grid">
            <div className="impact-item">
              <span className="impact-value">{data.business_impact.estimated_roi}x</span>
              <span className="impact-label">Estimated ROI</span>
            </div>
            <div className="impact-item">
              <span className="impact-value">{data.business_impact.cost_savings_potential}</span>
              <span className="impact-label">Cost Savings</span>
            </div>
            <div className="impact-item">
              <span className="impact-value">{data.business_impact.performance_gain}</span>
              <span className="impact-label">Performance Gain</span>
            </div>
          </div>
        </div>
      )}

      <style jsx>{`
        .tradeoff-container {
          max-width: 900px;
          margin: 0 auto;
          padding: 2rem;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        .controls-panel {
          display: flex;
          flex-wrap: wrap;
          gap: 1.5rem;
          padding: 1.5rem;
          background: #f5f5f5;
          border-radius: 8px;
          margin-bottom: 2rem;
          align-items: flex-end;
        }

        .control-group {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .control-group label {
          font-size: 0.875rem;
          color: #666;
        }

        .control-group input[type="range"] {
          width: 150px;
        }

        .control-group select {
          padding: 0.5rem;
          border: 1px solid #ddd;
          border-radius: 4px;
        }

        .analyze-btn {
          padding: 0.75rem 1.5rem;
          background: #2196f3;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-weight: 500;
        }

        .analyze-btn:disabled {
          opacity: 0.7;
        }

        .error-message {
          background: #ffebee;
          color: #c62828;
          padding: 1rem;
          border-radius: 4px;
          margin-bottom: 1rem;
        }

        .chart-container {
          background: white;
          padding: 1rem;
          border-radius: 8px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
          margin-bottom: 2rem;
        }

        .axis-label {
          font-size: 12px;
          fill: #666;
        }

        .point-label {
          font-size: 11px;
          fill: #333;
          font-weight: 500;
        }

        .badge {
          font-size: 16px;
          fill: #ffc107;
        }

        .chart-legend {
          display: flex;
          justify-content: center;
          gap: 2rem;
          margin-top: 1rem;
          padding-top: 1rem;
          border-top: 1px solid #eee;
        }

        .legend-item {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.875rem;
          color: #666;
        }

        .dot {
          width: 12px;
          height: 12px;
          border-radius: 50%;
        }

        .dot.recommended {
          background: #4caf50;
        }

        .dot.option {
          background: #2196f3;
        }

        .line.pareto {
          width: 24px;
          height: 3px;
          background: #2196f3;
          border-style: dashed;
        }

        .option-details {
          background: #f5f5f5;
          padding: 1.5rem;
          border-radius: 8px;
          margin-bottom: 2rem;
        }

        .details-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 1rem;
          margin-top: 1rem;
        }

        .detail-card {
          background: white;
          padding: 1rem;
          border-radius: 8px;
          text-align: center;
        }

        .detail-card h3 {
          font-size: 0.875rem;
          color: #666;
          margin: 0 0 0.5rem;
        }

        .detail-card .value {
          font-size: 1.5rem;
          font-weight: bold;
          color: #333;
          margin: 0;
        }

        .detail-card .label {
          font-size: 0.75rem;
          color: #999;
          margin: 0;
        }

        .business-impact {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          padding: 2rem;
          border-radius: 8px;
        }

        .business-impact h2 {
          margin: 0 0 1.5rem;
        }

        .impact-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 1rem;
        }

        .impact-item {
          text-align: center;
        }

        .impact-value {
          display: block;
          font-size: 2rem;
          font-weight: bold;
        }

        .impact-label {
          font-size: 0.875rem;
          opacity: 0.9;
        }
      `}</style>
    </div>
  );
};

export default PerformanceTradeoffVisualization;

