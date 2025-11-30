# Integration Examples

Dashboard integration examples for the AI Agent Performance Intelligence System.

**Course:** DATA 230 (Data Visualization) at SJSU

## Contents

### JavaScript API Client
`js/StrategicMLClient.js` - Full-featured API client with:
- Automatic retry logic
- Request timeout handling
- All API endpoints covered
- TypeScript-compatible JSDoc annotations

```javascript
import { StrategicMLClient } from './js/StrategicMLClient';

const client = new StrategicMLClient('http://localhost:8000');

// Get optimal configuration
const config = await client.getOptimalConfiguration({
  task_complexity: 7,
  autonomy_level: 6,
  min_success_rate: 0.85
});

// Get risk assessment
const risk = await client.getFailureRisk({
  agent_id: 'AG_001',
  success_rate: 0.75,
  accuracy_score: 0.80,
  efficiency_score: 0.70,
  execution_time_seconds: 5.0,
  response_latency_ms: 300,
  memory_usage_mb: 350,
  cpu_usage_percent: 60
});
```

### React Components

#### AgentOptimizationWizard.jsx
Multi-step wizard for configuring AI agent deployments:
- Task requirements input
- Performance constraints
- Optimization execution
- Results visualization

#### PerformanceTradeoffVisualization.jsx
Interactive Pareto frontier visualization:
- Cost vs performance scatter plot
- Pareto frontier line
- Interactive option selection
- Business impact display

#### RiskAssessmentDashboard.jsx
Real-time risk monitoring dashboard:
- Agent selection
- Metric sliders
- Risk gauge visualization
- Heat map display
- Mitigation recommendations

### Python Visualization Templates
`visualizations/chart_templates.py` - Plotly-based templates:

1. **Pareto Frontier Chart** - Cost-performance tradeoff visualization
2. **Risk Heat Map** - Multi-dimensional risk display
3. **Performance Trend Projection** - Historical trends with forecasting
4. **Configuration Impact Graph** - Radar chart for multi-metric comparison
5. **Agent Dashboard** - Comprehensive single-agent view

```python
from chart_templates import create_pareto_frontier_chart, create_risk_heatmap

# Create Pareto chart
fig = create_pareto_frontier_chart(
    data=pareto_df,
    cost_col='cost',
    performance_col='performance',
    recommended_col='is_recommended'
)
fig.show()
```

## Quick Start

### Using React Components

```bash
# Install dependencies
npm install react plotly.js react-plotly.js

# Import component
import AgentOptimizationWizard from './examples/react/AgentOptimizationWizard';

# Use in your app
<AgentOptimizationWizard />
```

### Using JavaScript Client

```html
<script src="examples/js/StrategicMLClient.js"></script>
<script>
  const client = new StrategicMLClient('http://localhost:8000');
  client.healthCheck().then(console.log);
</script>
```

### Using Python Templates

```python
# Install plotly
pip install plotly pandas numpy

# Import and use
from examples.visualizations.chart_templates import create_agent_dashboard

fig = create_agent_dashboard({
    'agent_id': 'AG_001',
    'success_rate': 0.85,
    'accuracy_score': 0.88,
    'efficiency_score': 0.75,
    'cost_efficiency': 72.5,
    'risk_score': 25,
    'cpu_usage_percent': 45,
    'memory_usage_mb': 350,
    'response_latency_ms': 180,
    'error_recovery_rate': 0.80
})
fig.show()
```

## Tableau/Power BI Integration

The API endpoints return JSON that can be consumed by:
- Tableau Web Data Connector
- Power BI REST API connector
- Custom dashboard integrations

Example Power BI query:

```m
let
    Source = Json.Document(Web.Contents("http://localhost:8000/v1/agent-recommendation-engine", [
        Headers=[#"Content-Type"="application/json"],
        Content=Text.ToBinary("{""task_complexity"": 7, ""autonomy_level"": 6}")
    ])),
    recommendations = Source[recommendations]
in
    recommendations
```

## Environment Variables

Set these in your frontend application:

```env
REACT_APP_API_URL=http://localhost:8000
```

