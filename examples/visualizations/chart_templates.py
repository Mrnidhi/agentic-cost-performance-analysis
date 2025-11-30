"""
Visualization Templates for AI Agent Performance Intelligence System
Course: DATA 230 (Data Visualization) at SJSU

Templates for:
- Pareto frontier charts
- Risk heat maps
- Performance trend projections
- Configuration impact graphs
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


# =============================================================================
# 1. Pareto Frontier Chart
# =============================================================================

def create_pareto_frontier_chart(data: pd.DataFrame, 
                                  cost_col: str = 'cost',
                                  performance_col: str = 'performance',
                                  label_col: str = 'option_name',
                                  recommended_col: str = None) -> go.Figure:
    """
    Create an interactive Pareto frontier visualization.
    
    Args:
        data: DataFrame with cost and performance data
        cost_col: Column name for cost values
        performance_col: Column name for performance values
        label_col: Column name for point labels
        recommended_col: Column name indicating recommended option (boolean)
    
    Returns:
        Plotly figure object
    """
    # Sort by cost for frontier line
    sorted_data = data.sort_values(cost_col)
    
    # Create figure
    fig = go.Figure()
    
    # Add Pareto frontier line
    fig.add_trace(go.Scatter(
        x=sorted_data[cost_col],
        y=sorted_data[performance_col],
        mode='lines',
        name='Pareto Frontier',
        line=dict(color='#2196f3', width=2, dash='dash'),
        hoverinfo='skip'
    ))
    
    # Add data points
    if recommended_col and recommended_col in data.columns:
        # Recommended points
        recommended = data[data[recommended_col] == True]
        non_recommended = data[data[recommended_col] == False]
        
        fig.add_trace(go.Scatter(
            x=recommended[cost_col],
            y=recommended[performance_col],
            mode='markers+text',
            name='Recommended',
            marker=dict(size=16, color='#4caf50', symbol='star'),
            text=recommended[label_col],
            textposition='top center',
            hovertemplate='<b>%{text}</b><br>Cost: %{x:.4f}<br>Performance: %{y:.2%}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=non_recommended[cost_col],
            y=non_recommended[performance_col],
            mode='markers+text',
            name='Alternatives',
            marker=dict(size=12, color='#2196f3'),
            text=non_recommended[label_col],
            textposition='top center',
            hovertemplate='<b>%{text}</b><br>Cost: %{x:.4f}<br>Performance: %{y:.2%}<extra></extra>'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=data[cost_col],
            y=data[performance_col],
            mode='markers+text',
            name='Options',
            marker=dict(size=12, color='#2196f3'),
            text=data[label_col] if label_col in data.columns else None,
            textposition='top center',
            hovertemplate='<b>%{text}</b><br>Cost: %{x:.4f}<br>Performance: %{y:.2%}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title='Cost-Performance Pareto Frontier',
        xaxis_title='Cost (cents/task)',
        yaxis_title='Performance Index',
        template='plotly_white',
        hovermode='closest',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    # Add shaded feasible region
    fig.add_vrect(
        x0=data[cost_col].min() * 0.9,
        x1=data[cost_col].max() * 0.5,
        fillcolor='rgba(76, 175, 80, 0.1)',
        layer='below',
        line_width=0,
        annotation_text='Budget-Friendly Zone',
        annotation_position='top left'
    )
    
    return fig


# =============================================================================
# 2. Risk Heat Map
# =============================================================================

def create_risk_heatmap(data: pd.DataFrame,
                        group_cols: list = ['agent_type', 'deployment_environment'],
                        risk_col: str = 'risk_score') -> go.Figure:
    """
    Create a risk heat map showing risk levels across dimensions.
    
    Args:
        data: DataFrame with risk data
        group_cols: Columns to use for x and y axes
        risk_col: Column containing risk scores
    
    Returns:
        Plotly figure object
    """
    # Pivot data for heatmap
    pivot_data = data.pivot_table(
        values=risk_col,
        index=group_cols[0],
        columns=group_cols[1],
        aggfunc='mean'
    )
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale=[
            [0, '#4caf50'],      # Low risk - Green
            [0.25, '#8bc34a'],
            [0.5, '#ffeb3b'],    # Medium risk - Yellow
            [0.75, '#ff9800'],
            [1, '#f44336']       # High risk - Red
        ],
        colorbar=dict(
            title='Risk Score',
            tickvals=[0, 25, 50, 75, 100],
            ticktext=['Low', 'Low-Med', 'Medium', 'Med-High', 'High']
        ),
        hovertemplate='%{y}<br>%{x}<br>Risk: %{z:.1f}<extra></extra>'
    ))
    
    # Add annotations
    for i, row in enumerate(pivot_data.index):
        for j, col in enumerate(pivot_data.columns):
            value = pivot_data.iloc[i, j]
            if not np.isnan(value):
                fig.add_annotation(
                    x=col,
                    y=row,
                    text=f'{value:.0f}',
                    showarrow=False,
                    font=dict(
                        color='white' if value > 50 else 'black',
                        size=12
                    )
                )
    
    fig.update_layout(
        title='Risk Heat Map by Agent Type and Environment',
        xaxis_title=group_cols[1].replace('_', ' ').title(),
        yaxis_title=group_cols[0].replace('_', ' ').title(),
        template='plotly_white'
    )
    
    return fig


# =============================================================================
# 3. Performance Trend Projection
# =============================================================================

def create_performance_trend_chart(historical_data: pd.DataFrame,
                                   date_col: str = 'date',
                                   performance_col: str = 'performance_index',
                                   projection_days: int = 30) -> go.Figure:
    """
    Create performance trend chart with future projections.
    
    Args:
        historical_data: DataFrame with historical performance data
        date_col: Column name for dates
        performance_col: Column name for performance values
        projection_days: Number of days to project forward
    
    Returns:
        Plotly figure object
    """
    # Ensure date column is datetime
    df = historical_data.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    # Calculate trend using simple linear regression
    x = np.arange(len(df))
    y = df[performance_col].values
    
    # Fit linear trend
    coeffs = np.polyfit(x, y, 1)
    trend_line = np.poly1d(coeffs)
    
    # Generate projection dates
    last_date = df[date_col].max()
    projection_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=projection_days,
        freq='D'
    )
    
    # Calculate projections
    projection_x = np.arange(len(df), len(df) + projection_days)
    projections = trend_line(projection_x)
    
    # Calculate confidence interval (simple approach)
    std = np.std(y - trend_line(x))
    upper_bound = projections + 1.96 * std
    lower_bound = projections - 1.96 * std
    
    # Create figure
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[performance_col],
        mode='lines+markers',
        name='Historical',
        line=dict(color='#2196f3', width=2),
        marker=dict(size=6)
    ))
    
    # Trend line
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=trend_line(x),
        mode='lines',
        name='Trend',
        line=dict(color='#666', width=1, dash='dash')
    ))
    
    # Projection
    fig.add_trace(go.Scatter(
        x=projection_dates,
        y=projections,
        mode='lines',
        name='Projection',
        line=dict(color='#4caf50', width=2, dash='dot')
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=list(projection_dates) + list(projection_dates[::-1]),
        y=list(upper_bound) + list(lower_bound[::-1]),
        fill='toself',
        fillcolor='rgba(76, 175, 80, 0.2)',
        line=dict(color='rgba(0,0,0,0)'),
        name='95% Confidence',
        hoverinfo='skip'
    ))
    
    # Add vertical line at current date
    fig.add_vline(
        x=last_date,
        line_dash='dash',
        line_color='#999',
        annotation_text='Today',
        annotation_position='top'
    )
    
    fig.update_layout(
        title='Performance Trend & Projection',
        xaxis_title='Date',
        yaxis_title='Performance Index',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    return fig


# =============================================================================
# 4. Configuration Impact Graph
# =============================================================================

def create_configuration_impact_chart(data: pd.DataFrame,
                                      config_col: str = 'configuration',
                                      metrics: list = None) -> go.Figure:
    """
    Create radar/spider chart showing configuration impact on multiple metrics.
    
    Args:
        data: DataFrame with configuration and metric data
        config_col: Column name for configuration labels
        metrics: List of metric columns to display
    
    Returns:
        Plotly figure object
    """
    if metrics is None:
        metrics = ['success_rate', 'accuracy', 'efficiency', 'cost_efficiency', 'reliability']
    
    # Normalize metrics to 0-1 scale
    normalized_data = data.copy()
    for metric in metrics:
        if metric in normalized_data.columns:
            min_val = normalized_data[metric].min()
            max_val = normalized_data[metric].max()
            if max_val > min_val:
                normalized_data[metric] = (normalized_data[metric] - min_val) / (max_val - min_val)
    
    # Create figure
    fig = go.Figure()
    
    # Color palette
    colors = px.colors.qualitative.Set2
    
    # Add trace for each configuration
    for idx, (_, row) in enumerate(normalized_data.iterrows()):
        values = [row.get(m, 0) for m in metrics]
        values.append(values[0])  # Close the polygon
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill='toself',
            fillcolor=f'rgba({",".join(str(int(c)) for c in px.colors.hex_to_rgb(colors[idx % len(colors)]))},0.2)',
            line=dict(color=colors[idx % len(colors)], width=2),
            name=row.get(config_col, f'Config {idx + 1}')
        ))
    
    fig.update_layout(
        title='Configuration Impact Analysis',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=['20%', '40%', '60%', '80%', '100%']
            )
        ),
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5
        )
    )
    
    return fig


# =============================================================================
# 5. Multi-Metric Dashboard
# =============================================================================

def create_agent_dashboard(agent_data: dict) -> go.Figure:
    """
    Create a comprehensive dashboard for a single agent.
    
    Args:
        agent_data: Dictionary with agent metrics
    
    Returns:
        Plotly figure object with subplots
    """
    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
            [{'type': 'bar', 'colspan': 2}, None, {'type': 'pie'}]
        ],
        subplot_titles=['Success Rate', 'Cost Efficiency', 'Risk Score', 
                       'Performance Breakdown', '', 'Resource Usage']
    )
    
    # Success Rate Gauge
    fig.add_trace(go.Indicator(
        mode='gauge+number',
        value=agent_data.get('success_rate', 0) * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': 'Success Rate'},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': '#4caf50'},
            'steps': [
                {'range': [0, 50], 'color': '#ffebee'},
                {'range': [50, 80], 'color': '#fff3e0'},
                {'range': [80, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 2},
                'thickness': 0.75,
                'value': 70
            }
        }
    ), row=1, col=1)
    
    # Cost Efficiency
    fig.add_trace(go.Indicator(
        mode='number+delta',
        value=agent_data.get('cost_efficiency', 0),
        delta={'reference': 50, 'relative': True},
        title={'text': 'Cost Efficiency'},
        number={'suffix': ' ratio'}
    ), row=1, col=2)
    
    # Risk Score
    fig.add_trace(go.Indicator(
        mode='gauge+number',
        value=agent_data.get('risk_score', 0),
        title={'text': 'Risk Score'},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': '#f44336'},
            'steps': [
                {'range': [0, 30], 'color': '#e8f5e9'},
                {'range': [30, 70], 'color': '#fff3e0'},
                {'range': [70, 100], 'color': '#ffebee'}
            ]
        }
    ), row=1, col=3)
    
    # Performance Breakdown Bar Chart
    metrics = ['Accuracy', 'Efficiency', 'Reliability', 'Speed']
    values = [
        agent_data.get('accuracy_score', 0.8) * 100,
        agent_data.get('efficiency_score', 0.75) * 100,
        agent_data.get('error_recovery_rate', 0.7) * 100,
        max(0, 100 - agent_data.get('response_latency_ms', 200) / 10)
    ]
    
    fig.add_trace(go.Bar(
        x=metrics,
        y=values,
        marker_color=['#2196f3', '#4caf50', '#ff9800', '#9c27b0'],
        text=[f'{v:.0f}%' for v in values],
        textposition='outside'
    ), row=2, col=1)
    
    # Resource Usage Pie
    fig.add_trace(go.Pie(
        labels=['CPU', 'Memory', 'Available'],
        values=[
            agent_data.get('cpu_usage_percent', 50),
            agent_data.get('memory_usage_mb', 300) / 10,
            100 - agent_data.get('cpu_usage_percent', 50) - agent_data.get('memory_usage_mb', 300) / 10
        ],
        marker_colors=['#f44336', '#ff9800', '#e0e0e0'],
        hole=0.4
    ), row=2, col=3)
    
    fig.update_layout(
        height=600,
        title_text=f"Agent Dashboard: {agent_data.get('agent_id', 'Unknown')}",
        template='plotly_white',
        showlegend=False
    )
    
    return fig


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == '__main__':
    # Sample data for testing
    pareto_data = pd.DataFrame({
        'option_name': ['Budget', 'Balanced', 'Premium'],
        'cost': [0.008, 0.012, 0.018],
        'performance': [0.65, 0.78, 0.88],
        'recommended': [False, True, False]
    })
    
    # Create and show Pareto chart
    pareto_fig = create_pareto_frontier_chart(
        pareto_data, 
        cost_col='cost',
        performance_col='performance',
        label_col='option_name',
        recommended_col='recommended'
    )
    pareto_fig.show()
    
    print("Visualization templates loaded successfully!")

