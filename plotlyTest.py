import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from Asset_Modeling.Asset_class import asset_BS
from Options.Book_class import Book
from Options.Options_class import Option_eu

# Simulated Asset and Option Instances
stock = asset_BS(100, 0, "AAPL")
option = Option_eu(10, 'Call EU', stock, 100, 30 / 365, 0.05, 0.25)
option2 = Option_eu(-10, 'Call EU', stock, 100.1, 5 / 365, 0.05, 0.25)

book = Book([option, option2])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Layout
app.layout = dbc.Container([
    html.H1("Option Trading Dashboard", className='text-center mb-5',
            style={'font-size': '3em', 'font-weight': 'bold'}),

    dbc.Row([
        dbc.Col([
            html.Label("Select Option"),
            dcc.Dropdown(
                id='option-selector',
                options=[{'label': 'Book', 'value': 'book'}] +
                        [{'label': f'Option {i + 1}: {opt.type}', 'value': i} for i, opt in enumerate(book.basket)],
                value=0
            )
        ], width=6)
    ], className='mb-4'),

    html.H3("Greeks", className='text-center mt-4'),
    dbc.Row(id='greek-kpi-display', className='mb-5'),

    dbc.Row([
        dbc.Col([
            html.Label("Underlying Price"),
            dcc.Slider(
                id='price-slider',
                min=50, max=150, step=1, value=option.asset.St,
                marks={i: str(i) for i in range(50, 151, 10)}
            )
        ], width=6),
        dbc.Col([
            html.Label("Volatility"),
            dcc.Slider(
                id='vol-slider',
                min=0.1, max=1.0, step=0.01, value=option.sigma,
                marks={i / 10: f'{i / 10}' for i in range(1, 11)}
            )
        ], width=6)
    ], className='mb-5'),

    # Greeks Section
    html.Div([
        html.H3("Greek Plots", className='text-center mt-4'),
        dbc.Row([
            dbc.Col(dcc.Graph(id='delta-graph'), width=6),
            dbc.Col(dcc.Graph(id='gamma-graph'), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='vega-graph'), width=6),
            dbc.Col(dcc.Graph(id='theta-graph'), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='vanna-graph'), width=6),
            dbc.Col(dcc.Graph(id='volga-graph'), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='speed-graph'), width=6)
        ])
    ]),

    # PnL Section
    html.Div([
        html.H3("PnL Decomposition", className='text-center mt-5'),
        dbc.Row(id='pnl-kpi-display', className='mb-5'),
        dbc.Row([
            dbc.Col(dcc.Graph(id='pnl-graph'), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='delta-pnl-graph'), width=6),
            dbc.Col(dcc.Graph(id='gamma-pnl-graph'), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='third-order-pnl-graph'), width=6)
        ])
    ])
], fluid=True)


# KPI Card Component
def create_kpi_card(title, value, color):
    return dbc.Card(
        dbc.CardBody([
            html.H4(title, className='card-title', style={'color': color}),
            html.H2(f"{value:.2f}", className='card-text')
        ]),
        className="shadow-sm text-center",
        style={'border-radius': '10px'}
    )

# Plotting Function for Greeks

def plot_greek(greek_method, price, current_greek, title):
    curve = greek_method(plot=False)
    range_st = curve.index

    fig = go.Figure(data=go.Scatter(x=range_st, y=curve, mode='lines', name=f'{title} Curve'))
    fig.add_trace(go.Scatter(x=[price], y=[current_greek],
                             mode='markers', line=dict(dash='dash', color="red"), name=f'Current {title}'))
    fig.update_layout(title=f'{title} Curve', xaxis_title='Underlying Price', yaxis_title=title)
    return fig

@app.callback(
    [Output('delta-graph', 'figure'),
     Output('gamma-graph', 'figure'),
     Output('vega-graph', 'figure'),
     Output('theta-graph', 'figure'),
     Output('vanna-graph', 'figure'),
     Output('volga-graph', 'figure'),
     Output('speed-graph', 'figure'),
     Output('pnl-graph', 'figure'),
     Output('delta-pnl-graph', 'figure'),
     Output('gamma-pnl-graph', 'figure'),
     Output('third-order-pnl-graph', 'figure'),
     Output('greek-kpi-display', 'children'),
     Output('pnl-kpi-display', 'children')],
    [Input('option-selector', 'value'),
     Input('price-slider', 'value'),
     Input('vol-slider', 'value')]
)
def update_graphs(selected_option, price, vol):
    if selected_option == 'book':
        option = book
    else:
        option = book.basket[selected_option]

    option.asset.St = price
    option.sigma = vol

    # Greeks (KPI values)
    current_delta = option.Delta_DF()
    current_gamma = option.Gamma_DF()
    current_vega = option.Vega_DF()
    current_theta = option.Theta_DF()
    current_vanna = option.Vanna_DF()
    current_volga = option.Volga_DF()
    current_speed = option.Speed_DF()

    # PnL Breakdown
    delta_pnl = option.Delta_Pnl(plot=False).loc[price]
    gamma_pnl = option.Gamma_Pnl(plot=False).loc[price]
    third_order_pnl = option.Third_Order_Pnl(plot=False).loc[price]
    n_order_pnl = option.nOrderPnl(plot=False).loc[price]
    # Generate Greek Graphs
    delta_fig = plot_greek(option.DeltaRisk, price, current_delta, 'Delta')
    gamma_fig = plot_greek(option.GammaRisk, price, current_gamma, 'Gamma')
    vega_fig = plot_greek(option.VegaRisk, price, current_vega, 'Vega')
    theta_fig = plot_greek(option.ThetaRisk, price, current_theta, 'Theta')

    vanna_fig = plot_greek(option.VannaRisk, price, current_vanna, 'Vanna')
    volga_fig = plot_greek(option.VolgaRisk, price, current_volga, 'Volga')
    speed_fig = plot_greek(option.SpeedRisk, price, current_speed, 'Speed')

    pnl_curve = option.PnlRisk(plot=False)
    pnl_fig = go.Figure(data=go.Scatter(x=pnl_curve.index, y=pnl_curve.values, mode='lines', name='PnL Curve'))
    pnl_fig.update_layout(title='PnL Curve', xaxis_title='Underlying Price', yaxis_title='PnL')

    delta_pnl_fig = plot_greek(option.Delta_Pnl, price, delta_pnl, 'Delta PnL')
    gamma_pnl_fig = plot_greek(option.Gamma_Pnl, price, gamma_pnl, 'Gamma PnL')
    third_order_pnl_fig = plot_greek(option.Third_Order_Pnl, price, third_order_pnl, '3rd Order PnL')

    # Create KPI Display for Greeks
    greek_display = dbc.Row([
        dbc.Col(create_kpi_card("Delta", current_delta, "blue"), width=3),
        dbc.Col(create_kpi_card("Gamma", current_gamma, "orange"), width=3),
        dbc.Col(create_kpi_card("Vega", current_vega, "purple"), width=3),
        dbc.Col(create_kpi_card("Theta", current_theta, "red"), width=3),
        dbc.Col(create_kpi_card("Vanna", current_vanna, "green"), width=3),
        dbc.Col(create_kpi_card("Volga", current_volga, "teal"), width=3),
        dbc.Col(create_kpi_card("Speed", current_speed, "darkred"), width=3)
    ])

    # Create KPI Display for PnL
    pnl_display = dbc.Row([
        dbc.Col(create_kpi_card("Delta PnL", delta_pnl, "green"), width=3),
        dbc.Col(create_kpi_card("Gamma PnL", gamma_pnl, "teal"), width=3),
        dbc.Col(create_kpi_card("3rd Order PnL", third_order_pnl, "orange"), width=3),
        dbc.Col(create_kpi_card("N-Order PnL", n_order_pnl, "goldenrod"), width=3)
    ])

    return (delta_fig, gamma_fig, vega_fig, theta_fig,
            vanna_fig, volga_fig, speed_fig, pnl_fig,
            delta_pnl_fig, gamma_pnl_fig, third_order_pnl_fig,
            greek_display, pnl_display)
if __name__ == '__main__':
    app.run_server(debug=True)


