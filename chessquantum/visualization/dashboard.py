"""
Publication-Ready Visualization Dashboard for Quantum-Chess HFT Framework.
Built with Plotly Dash for real-time and static visualizations.
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import json

# Initialize Dash app
app = dash.Dash(__name__, title="Quantum-Chess HFT Dashboard")

# Color scheme for publication-ready visuals
COLORS = {
    'background': '#0a0a0a',      # Dark background
    'text': '#e0e0e0',            # Light gray text
    'primary': '#00b4d8',         # Cyan blue
    'secondary': '#ffb703',        # Amber
    'success': '#06ffa5',          # Mint green
    'danger': '#ff4d4d',           # Red
    'grid': '#2a2a2a',             # Dark gray grid
    'bid': '#2ecc71',              # Green for bids
    'ask': '#e74c3c'               # Red for asks
}

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Quantum-Chess HFT Framework", 
                style={'color': COLORS['primary'], 'marginBottom': '5px'}),
        html.P("Research-Grade Trading Intelligence System",
               style={'color': COLORS['text'], 'fontSize': '14px'})
    ], style={'textAlign': 'center', 'padding': '20px', 'borderBottom': f'1px solid {COLORS["grid"]}'}),
    
    # Main content
    html.Div([
        # Left panel: Real-time visualizations
        html.Div([
            html.H3("Market Heatmap", style={'color': COLORS['text']}),
            dcc.Graph(id='market-heatmap', config={'displayModeBar': False}),
            
            html.H3("Quantum Decision Tree", style={'color': COLORS['text'], 'marginTop': '20px'}),
            dcc.Graph(id='decision-tree', config={'displayModeBar': False}),
            
            html.H3("Action Probability Distribution", style={'color': COLORS['text'], 'marginTop': '20px'}),
            dcc.Graph(id='action-probs', config={'displayModeBar': False})
        ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
        
        # Right panel: Analytics
        html.Div([
            html.H3("Risk-Reward Surface", style={'color': COLORS['text']}),
            dcc.Graph(id='risk-surface', config={'displayModeBar': False}),
            
            html.H3("Strategy Evolution", style={'color': COLORS['text'], 'marginTop': '20px'}),
            dcc.Graph(id='strategy-evolution', config={'displayModeBar': False}),
            
            html.H3("Latency Profile", style={'color': COLORS['text'], 'marginTop': '20px'}),
            dcc.Graph(id='latency-profile', config={'displayModeBar': False})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'})
    ]),
    
    # Control panel
    html.Div([
        html.Button("Update", id="update-button", n_clicks=0,
                   style={'backgroundColor': COLORS['primary'], 'color': 'white', 
                          'border': 'none', 'padding': '10px 20px', 'margin': '10px',
                          'cursor': 'pointer'}),
        html.Button("Export PNG", id="export-button", n_clicks=0,
                   style={'backgroundColor': COLORS['secondary'], 'color': 'white',
                          'border': 'none', 'padding': '10px 20px', 'margin': '10px',
                          'cursor': 'pointer'}),
        dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
    ], style={'textAlign': 'center', 'padding': '20px', 'borderTop': f'1px solid {COLORS["grid"]}'})
], style={'backgroundColor': COLORS['background'], 'fontFamily': 'Helvetica, Roboto, sans-serif'})


@app.callback(
    Output('market-heatmap', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('update-button', 'n_clicks')]
)
def update_market_heatmap(n_intervals, n_clicks):
    """Create dynamic order book heatmap."""
    # Generate synthetic order book data
    price_levels = np.arange(95, 105, 0.5)
    bid_volumes = np.exp(-((price_levels - 99.5) ** 2) / 2) * 1000
    ask_volumes = np.exp(-((price_levels - 100.5) ** 2) / 2) * 1000
    
    fig = go.Figure()
    
    # Bid side (negative for left side)
    fig.add_trace(go.Bar(
        x=-bid_volumes,
        y=price_levels,
        orientation='h',
        name='Bids',
        marker_color=COLORS['bid'],
        marker_line_width=0,
        opacity=0.8
    ))
    
    # Ask side
    fig.add_trace(go.Bar(
        x=ask_volumes,
        y=price_levels,
        orientation='h',
        name='Asks',
        marker_color=COLORS['ask'],
        marker_line_width=0,
        opacity=0.8
    ))
    
    fig.update_layout(
        title="Limit Order Book Depth",
        xaxis_title="Volume",
        yaxis_title="Price Level",
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font_color=COLORS['text'],
        xaxis=dict(gridcolor=COLORS['grid']),
        yaxis=dict(gridcolor=COLORS['grid']),
        bargap=0.1,
        bargroupgap=0.05,
        legend=dict(x=0.9, y=1.1, orientation='h')
    )
    
    return fig


@app.callback(
    Output('decision-tree', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_decision_tree(n_intervals):
    """Create quantum decision tree visualization (Sankey diagram)."""
    # Simplified tree structure
    labels = ["Market State", "Action A", "Action B", "Action C", 
              "Outcome A1", "Outcome A2", "Outcome B1", "Outcome B2"]
    
    # Source, target, value for Sankey
    source = [0, 0, 0, 1, 1, 2, 2]
    target = [1, 2, 3, 4, 5, 6, 7]
    value = [0.4, 0.35, 0.25, 0.6, 0.4, 0.55, 0.45]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color=COLORS['grid'], width=0.5),
            label=labels,
            color=[COLORS['primary']] + [COLORS['secondary']] * 3 + [COLORS['success']] * 4
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=[f'rgba(0, 180, 216, {v})' for v in value]
        )
    )])
    
    fig.update_layout(
        title="Quantum Decision Tree (Action Probabilities)",
        font=dict(size=12, color=COLORS['text']),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background']
    )
    
    return fig


@app.callback(
    Output('action-probs', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_action_probabilities(n_intervals):
    """Create action probability distribution bar chart."""
    actions = ['Sell (-5)', 'Sell (-4)', 'Sell (-3)', 'Sell (-2)', 'Sell (-1)',
               'Hold (0)',
               'Buy (+1)', 'Buy (+2)', 'Buy (+3)', 'Buy (+4)', 'Buy (+5)']
    
    # Simulate probabilities from quantum layer
    np.random.seed(n_intervals)
    probs = np.random.dirichlet(np.ones(11))
    
    colors = [COLORS['ask'] if i < 5 else COLORS['primary'] if i > 5 else COLORS['secondary'] 
              for i in range(11)]
    
    fig = go.Figure(data=[go.Bar(
        x=actions,
        y=probs,
        marker_color=colors,
        marker_line_width=0,
        opacity=0.8
    )])
    
    fig.update_layout(
        title="Action Probability Distribution P(A|S)",
        xaxis_title="Action (Price Offset in Ticks)",
        yaxis_title="Probability",
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font_color=COLORS['text'],
        xaxis=dict(gridcolor=COLORS['grid'], tickangle=45),
        yaxis=dict(gridcolor=COLORS['grid'])
    )
    
    return fig


@app.callback(
    Output('risk-surface', 'figure'),
    [Input('update-button', 'n_clicks')]
)
def update_risk_surface(n_clicks):
    """Create 3D risk-reward surface."""
    # Generate surface data
    position_risk = np.linspace(0, 1, 20)
    volatility = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(position_risk, volatility)
    
    # Expected profit as function of risk and volatility
    Z = 0.5 * np.exp(-2 * X) - 0.3 * Y**2 + 0.2 * np.sin(3 * X) * np.cos(2 * Y)
    
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))
    )])
    
    fig.update_layout(
        title="Risk-Reward Surface: Expected Profit",
        scene=dict(
            xaxis_title="Position Risk",
            yaxis_title="Market Volatility",
            zaxis_title="Expected Profit",
            xaxis=dict(gridcolor=COLORS['grid']),
            yaxis=dict(gridcolor=COLORS['grid']),
            zaxis=dict(gridcolor=COLORS['grid']),
            bgcolor=COLORS['background']
        ),
        paper_bgcolor=COLORS['background'],
        font_color=COLORS['text']
    )
    
    return fig


@app.callback(
    Output('strategy-evolution', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_strategy_evolution(n_intervals):
    """Create strategy evolution chart (Sharpe ratio over training)."""
    episodes = np.arange(0, 100, 5)
    
    # Simulate Sharpe ratio evolution with noise
    sharpe_baseline = 0.5 + 0.1 * np.log(episodes + 1)
    sharpe_qc = 0.8 + 0.25 * np.log(episodes + 1) + np.random.randn(len(episodes)) * 0.05
    sharpe_rl = 0.6 + 0.2 * np.log(episodes + 1) + np.random.randn(len(episodes)) * 0.08
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=episodes, y=sharpe_baseline,
        mode='lines+markers',
        name='Baseline Strategy',
        line=dict(color=COLORS['grid'], width=2),
        marker=dict(size=4)
    ))
    
    fig.add_trace(go.Scatter(
        x=episodes, y=sharpe_qc,
        mode='lines+markers',
        name='Quantum-Chess Framework',
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(size=4, symbol='triangle-up')
    ))
    
    fig.add_trace(go.Scatter(
        x=episodes, y=sharpe_rl,
        mode='lines+markers',
        name='RL-Only Baseline',
        line=dict(color=COLORS['secondary'], width=2),
        marker=dict(size=4, symbol='square')
    ))
    
    fig.update_layout(
        title="Strategy Evolution: Sharpe Ratio",
        xaxis_title="Training Episode",
        yaxis_title="Sharpe Ratio",
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font_color=COLORS['text'],
        xaxis=dict(gridcolor=COLORS['grid']),
        yaxis=dict(gridcolor=COLORS['grid']),
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig


@app.callback(
    Output('latency-profile', 'figure'),
    [Input('update-button', 'n_clicks')]
)
def update_latency_profile(n_clicks):
    """Create latency profile violin plot."""
    np.random.seed(42)
    
    # Simulate latency distributions (microseconds)
    qc_latency = np.random.gamma(2, 50, 500) + 50
    engine_latency = np.random.gamma(1.5, 30, 500) + 20
    rl_latency = np.random.gamma(1.2, 20, 500) + 15
    
    fig = go.Figure()
    
    fig.add_trace(go.Violin(
        y=qc_latency,
        x=['Quantum QAOA'] * 500,
        name='Quantum QAOA',
        box_visible=True,
        meanline_visible=True,
        fillcolor=COLORS['primary'],
        line_color=COLORS['primary']
    ))
    
    fig.add_trace(go.Violin(
        y=engine_latency,
        x=['Expectiminimax'] * 500,
        name='Expectiminimax',
        box_visible=True,
        meanline_visible=True,
        fillcolor=COLORS['secondary'],
        line_color=COLORS['secondary']
    ))
    
    fig.add_trace(go.Violin(
        y=rl_latency,
        x=['RL Agent'] * 500,
        name='RL Agent',
        box_visible=True,
        meanline_visible=True,
        fillcolor=COLORS['success'],
        line_color=COLORS['success']
    ))
    
    fig.update_layout(
        title="Decision Latency Profile (Microseconds)",
        xaxis_title="Component",
        yaxis_title="Latency (µs)",
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font_color=COLORS['text'],
        xaxis=dict(gridcolor=COLORS['grid']),
        yaxis=dict(gridcolor=COLORS['grid'])
    )
    
    return fig


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)