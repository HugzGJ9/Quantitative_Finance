# Dashboard/layout.py
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table

def create_layout(available_books, initial_book_name):
    return dbc.Container([
        html.H1("Option Dashboard (Imported Book)", className='text-center mb-4',
                style={'font-size': '2em', 'font-weight': 'bold'}),
        dbc.Row([
            dbc.Col([
                html.Label("Select Book to Import"),
                dcc.Dropdown(
                    id='book-selector',
                    options=[{'label': name, 'value': name} for name in available_books],
                    value=initial_book_name,
                    clearable=False
                )
            ], width=6),
            dbc.Col([
                html.H4(id='loaded-book-display', className='text-center')
            ], width=6)
        ], className='mb-4'),
        dcc.Tabs(id='tabs', value='dashboard', children=[
            dcc.Tab(label='Dashboard', value='dashboard', children=[
                dbc.Row([
                    dbc.Col([
                        html.Div(id='empty-book-alert', className='mb-3'),
                        html.Label("Select Option or Book"),
                        dcc.Dropdown(
                            id='option-selector',
                            options=[],
                            value='book'
                        )
                    ], width=6)
                ], className='mb-2'),
                html.Div([
                    html.H4("Positions in Book (MtM)", className='text-center mt-3'),
                    dash_table.DataTable(
                        id='position-table',
                        columns=[],
                        data=[],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'center', 'padding': '8px'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                    )
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H4("Run Mark-to-Market", className='text-center mb-3'),
                        dbc.Form([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Lot Size"),
                                    dbc.Input(
                                        id='mtm-lot-size',
                                        type='number',
                                        value=10000,
                                        min=1,
                                        step=1,
                                        disabled=True
                                    )
                                ], width=4),
                                dbc.Col([
                                    dbc.Button("Run MtM", id='run-mtm-button', color='primary', className='mt-4')
                                ], width=4)
                            ], className='mb-3'),
                            html.Div(id='mtm-feedback', className='mt-2')
                        ])
                    ], width=12)
                ], className='mb-4'),
                html.Div([
                    html.H4("Payoff", className='text-center mt-4'),
                    dcc.Graph(id='payoff-graph')
                ]),
                html.H4("Greeks", className='text-center mt-4'),
                html.Div(id='greek-kpi-display', className='mb-4'),
                html.Label("Underlying Price"),
                dcc.Slider(
                    id='price-slider',
                    min=80,
                    max=120,
                    step=0.1,
                    value=100,
                    marks={},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.Br(),
                html.Div(id='vol-sliders-container', className='mb-4'),
                html.H4("Greek Plots", className='text-center mt-4'),
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
                ]),
                html.H4("PnL Decomposition - IN PROGRESS", className='text-center mt-4'),
                html.Div(id='pnl-kpi-display'),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='pnl-graph'), width=6)
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='delta-pnl-graph'), width=6),
                    dbc.Col(dcc.Graph(id='gamma-pnl-graph'), width=6)
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='third-order-pnl-graph'), width=6)
                ]),
            ]),
            dcc.Tab(label='Book New Trade', value='booking', children=[
                html.H4("Book a New Trade", className='text-center mt-3'),
                dbc.Form([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Position"),
                            dcc.Dropdown(
                                id='new-trade-position',
                                options=[
                                    {'label': 'Long', 'value': 'long'},
                                    {'label': 'Short', 'value': 'short'}
                                ],
                                value='long'
                            )
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Type"),
                            dcc.Dropdown(
                                id='new-trade-type',
                                options=[
                                    {'label': 'Call European', 'value': 'Call EU'},
                                    {'label': 'Put European', 'value': 'Put EU'},
                                    {'label': 'Asset', 'value': 'Asset'}
                                ],
                                value='Call EU'
                            )
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Asset", className='fw-bold'),
                            dcc.Dropdown(
                                id='new-trade-asset',
                                options=[
                                    {'label': 'Nat Gas (NG=F)', 'value': 'NG=F'},
                                    {'label': 'Crude Oil (CL=F)', 'value': 'CL=F'},
                                    {'label': 'Gold (GC=F)', 'value': 'GC=F'},
                                    {'label': 'AAPL', 'value': 'AAPL'},
                                    {'label': 'TSLA', 'value': 'TSLA'}
                                ],
                                value='NG=F',  # default selection
                                clearable=False
                            )
                        ], width=3),

                        dbc.Col([
                            dbc.Label("Strike Price"),
                            dbc.Input(
                                id='new-trade-strike',
                                type='number',
                                value=100,
                                min=0
                            )
                        ], width=3),
                    ], className='mb-3'),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Maturity (days)"),
                            dbc.Input(
                                id='new-trade-maturity',
                                type='number',
                                value=30,
                                min=1
                            )
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Quantity"),
                            dbc.Input(
                                id='new-trade-quantity',
                                type='number',
                                value=1,
                                min=1
                            )
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Booked Price"),
                            dbc.Input(
                                id='new-trade-booked-price',
                                type='number',
                                value=0.0,
                                min=0.0,
                                step=0.01
                            )
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Volatility"),
                            dbc.Input(
                                id='new-trade-volatility',
                                type='number',
                                value=0.2,
                                min=0.0,
                                step=0.01
                            )
                        ], width=3),
                    ], className='mb-3'),
                    dbc.Button("Book Trade", id='book-trade-button', color='primary', className='mt-2'),
                    html.Div(id='booking-feedback', className='mt-3')
                ])
            ])
        ]),
        dcc.Store(id='current-book', data=initial_book_name)
    ], fluid=True)
