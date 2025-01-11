##########################################################
# dash_dashboard.py
# Enhanced to import a book from an Excel file, implement caching,
# handle HTTP 429 errors gracefully, set dynamic slider bounds,
# and add a booking tab for new trades.
##########################################################

import dash
from dash import dcc, html, Input, Output, State, dash_table, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import warnings
import copy
import os
from flask import Flask
from urllib.error import HTTPError

import Asset_Modeling.Asset_class
# --- Import your existing classes (ensure these are correctly defined in their respective modules)
from Asset_Modeling.Asset_class import asset_BS
from Options.Options_class import Option_eu
from Options.Book_class import Book
from Logger.Logger import mylogger
from Booking.Booking_Request import Booking_Request  # Import Booking_Request
from Volatility.Volatility import SMILE

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Initialize Flask server for Dash
server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.FLATLY])

##########################################################
# Initialize Cache Variables
##########################################################
from datetime import datetime, timedelta

last_fetch_time = None
cached_price = None
cache_duration = timedelta(minutes=5)  # Cache duration set to 5 minutes

##########################################################
# Function to Load Book from Excel with Caching and Error Handling
##########################################################
def load_book(book_name=None, smile_df=None):
    """
    Loads a Book from an Excel file with caching to minimize yfinance API calls.

    Parameters:
    - book_name (str): Name of the book file without extension.
    - smile_df (pd.DataFrame): Volatility smile DataFrame.

    Returns:
    - Book instance or None if an error occurs.
    """
    global last_fetch_time, cached_price
    booking_file_path = f"Booking/{book_name}.xlsx" if book_name else "Booking/Booking_history.xlsx"
    booking_file_sheet_name = 'histo_order'

    # Load booking data
    try:
        df = pd.read_excel(booking_file_path, sheet_name=booking_file_sheet_name)
    except FileNotFoundError:
        mylogger.logger.critical(f"Booking file not found: {booking_file_path}")
        return None
    except Exception as e:
        mylogger.logger.critical(f"An unexpected error occurred while reading the booking file: {e}")
        return None

    # Validate single asset
    unique_assets = df['asset'].dropna().unique()
    if len(unique_assets) != 1:
        mylogger.logger.critical('Multiple assets within the book.')
        return None
    asset_ticker = unique_assets[0]

    # Implement caching
    try:
        if last_fetch_time and datetime.now() - last_fetch_time < cache_duration:
            ng2_price = cached_price
            mylogger.logger.info("Using cached asset price.")
        else:
            ng2_data = yf.Ticker(asset_ticker)
            ng2_price = ng2_data.history(period='2d')['Close'].iloc[0]
            cached_price = ng2_price
            last_fetch_time = datetime.now()
            mylogger.logger.info("Fetched new asset price via yfinance.")
    except HTTPError as e:
        if e.code == 429:
            mylogger.logger.warning("HTTP 429: Too Many Requests. Using asset price from Excel.")
            # Use the asset price from Excel as a fallback
            try:
                ng2_price = df['asset price'].unique()[0]
            except Exception as ex:
                mylogger.logger.critical(f"Failed to retrieve asset price from Excel: {ex}")
                return None
        else:
            mylogger.logger.critical(f"HTTPError while fetching asset price: {e}")
            # Use the asset price from Excel as a fallback
            try:
                ng2_price = df['asset price'].unique()[0]
            except Exception as ex:
                mylogger.logger.critical(f"Failed to retrieve asset price from Excel: {ex}")
                return None
    except Exception as e:
        mylogger.logger.critical(f"An unexpected error occurred while fetching asset price: {e}")
        # Use the asset price from Excel as a fallback
        try:
            ng2_price = df['asset price'].unique()[0]
        except Exception as ex:
            mylogger.logger.critical(f"Failed to retrieve asset price from Excel: {ex}")
            return None

    # Initialize asset
    asset = asset_BS(ng2_price, 0, asset_ticker)

    # Process positions
    list_of_positions = []
    for i in range(len(df)):
        position = df.loc[i]
        if position.type.lower() == 'asset':
            asset.quantity += position.quantité
        else:
            try:
                delta = datetime.now().date() - pd.to_datetime(position['date heure']).date()
                time2matu = position.maturité - delta.total_seconds() / (24 * 3600)
                option = Option_eu(
                    position=position.quantité,
                    type=position.type,
                    asset=asset,
                    K=position.strike,
                    T=time2matu / 365,
                    r=0.1,  # Assuming a risk-free rate; adjust as needed
                    volatility_surface_df=smile_df,
                    use_vol_surface=True,
                    booked_price=position.SP  # Assuming 's-p' is booked_price
                )
                list_of_positions.append(option)
            except Exception as e:
                mylogger.logger.critical(f"Error processing position {i}: {e}")
                continue

    # Create and clean the book
    book = Book(list_of_positions)
    book.clean_basket()
    return book

##########################################################
# Helper function: Shift entire vol surface
##########################################################
def shift_vol_surface(opt_obj, shift_value):
    """
    Shift the entire implied vol surface in 'opt_obj' by 'shift_value'.
    E.g., shift_value=0.02 => all vols increase by 2 vol points.
    Then re-initialize the interpolator.
    """
    if not opt_obj.use_vol_surface or opt_obj.volatility_surface_df is None:
        return  # No-op if the option doesn't have a surface

    vol_cols = opt_obj.volatility_surface_df.columns[1:]  # skip 'Strike_percentage'
    for col in vol_cols:
        opt_obj.volatility_surface_df[col] += shift_value

    opt_obj.set_volatility_surface()

##########################################################
# 4) Load the Volatility Smile
##########################################################
try:
    mylogger.logger.info("Successfully loaded the volatility smile.")
except FileNotFoundError:
    mylogger.logger.critical("Volatility smile file not found: Volatility/Smile.xlsx")
    SMILE = None
except Exception as e:
    mylogger.logger.critical(f"An unexpected error occurred while loading volatility smile: {e}")
    SMILE = None

##########################################################
# 5) Load the Book
##########################################################
# --- Load the Book (default book name can be set here)
default_book_name = 'BookTest'  # Change as needed
book = load_book(default_book_name, smile_df=SMILE)
if book is None:
    # Handle the case where the book couldn't be loaded
    mylogger.logger.critical("Failed to load the book. Exiting dashboard.")
    exit()

# --- Calculate dynamic slider bounds based on asset price
if book.asset:
    asset_price = book.asset.St
    min_price = round(0.8 * asset_price, 2)
    max_price = round(1.2 * asset_price, 2)
    initial_price = asset_price
else:
    # Fallback values if no asset is present
    asset_price = 100
    min_price = 80
    max_price = 120
    initial_price = 100

# --- Define slider marks at significant points
marks = {
    round(0.8 * asset_price, 2): f"{round(0.8 * asset_price, 2)}",
    round(0.9 * asset_price, 2): f"{round(0.9 * asset_price, 2)}",
    round(asset_price, 2): f"{round(asset_price, 2)}",
    round(1.1 * asset_price, 2): f"{round(1.1 * asset_price, 2)}",
    round(1.2 * asset_price, 2): f"{round(1.2 * asset_price, 2)}"
}

# ============ Layout ============
app.layout = dbc.Container([

    html.H1("Option Dashboard (Imported Book)", className='text-center mb-4',
            style={'font-size': '2em', 'font-weight': 'bold'}),

    # -- Book Name Display
    dbc.Row([
        dbc.Col([
            html.H4(f"Loaded Book: {default_book_name}", className='text-center')
        ], width=12)
    ], className='mb-4'),

    # -- Tabs for Dashboard and Booking
    dcc.Tabs(id='tabs', value='dashboard', children=[
        dcc.Tab(label='Dashboard', value='dashboard', children=[
            # -- Option Selector
            dbc.Row([
                dbc.Col([
                    html.Label("Select Option or Book"),
                    dcc.Dropdown(
                        id='option-selector',
                        options=[{'label': 'Book (All)', 'value': 'book'}] +
                                [{'label': f'Option {i + 1}: {opt.type}', 'value': i}
                                 for i, opt in enumerate(book.basket)],
                        value='book'  # default
                    )
                ], width=6)
            ], className='mb-2'),

            # -- Positions Table
            html.Div([
                html.H4("Positions in Book", className='text-center mt-3'),
                dash_table.DataTable(
                    id='position-table',
                    columns=[
                        {"name": "Position", "id": "Position"},
                        {"name": "Quantity", "id": "Quantity"},
                        {"name": "Type", "id": "Type"},
                        {"name": "Strike", "id": "Strike"},
                        {"name": "Maturity (days)", "id": "Maturity (days)"},
                        {"name": "Underlying Asset", "id": "Underlying Asset"}
                    ],
                    data=[],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'center', 'padding': '8px'}
                )
            ]),

            # -- Payoff
            html.Div([
                html.H4("Payoff", className='text-center mt-4'),
                dcc.Graph(id='payoff-graph')
            ]),

            # -- Greeks KPI
            html.H4("Greeks", className='text-center mt-4'),
            html.Div(id='greek-kpi-display', className='mb-4'),

            # -- Underlying Price Slider
            html.Label("Underlying Price"),
            dcc.Slider(
                id='price-slider',
                min=min_price,
                max=max_price,
                step=(max_price - min_price) / 100,  # Adjust step based on range
                value=initial_price,  # Initialize with asset's current price
                marks=marks,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Br(),

            # -- Vol Sliders Container (Dynamically created)
            html.Div(id='vol-sliders-container', className='mb-4'),

            # -- Greek Plots
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

            # -- PnL
            html.H4("PnL Decomposition", className='text-center mt-4'),
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
                        dbc.Label("Asset"),
                        dcc.Input(
                            id='new-trade-asset',
                            type='text',
                            value=book.asset.name if book.asset else 'AAPL',
                            disabled=True  # Assuming single asset as per existing code
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

], fluid=True)

##########################################################
# Helper Functions
##########################################################
def create_kpi_card(title, value, color):
    """
    Creates a KPI card for displaying metrics.

    Parameters:
    - title (str): Title of the KPI.
    - value (float): Value of the KPI.
    - color (str): Color of the title text.

    Returns:
    - dbc.Card: A styled card component.
    """
    return dbc.Card(
        dbc.CardBody([
            html.H5(title, className='card-title', style={'color': color}),
            html.H2(f"{value:.2f}", className='card-text')
        ]),
        className="shadow-sm text-center mb-2",
        style={'border-radius': '10px'}
    )

def plot_method(method, price, current_value, title):
    """
    Plots the Greek or PnL curve and marks the current value.

    Parameters:
    - method (callable): Function to retrieve the curve data.
    - price (float): Current underlying price.
    - current_value (float): Current value of the Greek or PnL.
    - title (str): Title for the plot.

    Returns:
    - plotly.graph_objects.Figure: The generated plot.
    """
    try:
        curve = method(plot=False)  # Removed plot=False if not necessary
        if isinstance(curve, pd.Series):
            curve = curve.copy()
            curve.index = curve.index.astype(float)
            range_st = curve.index
            values = curve.values
        elif isinstance(curve, pd.DataFrame):
            curve = curve.copy()
            range_st = curve['Underlying Asset Price (St) move'].astype(float)
            values = curve['gains'].values
        else:
            return go.Figure().update_layout(title=f"No data for {title}")

        fig = go.Figure(data=go.Scatter(x=range_st, y=values, mode='lines', name=f'{title} Curve'))
        fig.add_trace(go.Scatter(
            x=[price], y=[current_value],
            mode='markers', marker=dict(color='red'), name=f'Current {title}'
        ))
        fig.update_layout(title=f'{title} Curve', xaxis_title='Underlying Price', yaxis_title=title)
        return fig
    except Exception as e:
        mylogger.logger.critical(f"Error plotting {title}: {e}")
        return go.Figure().update_layout(title=f"Error plotting {title}")

def safe_loc(series_or_df, key):
    """
    Safely retrieves a value from a Series or DataFrame.

    Parameters:
    - series_or_df (pd.Series or pd.DataFrame): The data structure to retrieve from.
    - key: The key/index to look up.

    Returns:
    - The retrieved value if key exists; otherwise, 0.
    """
    if isinstance(series_or_df, pd.Series) or isinstance(series_or_df, pd.DataFrame):
        if key in series_or_df.index:
            return series_or_df.loc[key]
    return 0

##########################################################
# 6) DYNAMIC Slider Callback:
#    Build one slider per relevant option if Book is selected,
#    or a single slider if a single option is selected.
##########################################################
@app.callback(
    Output('vol-sliders-container', 'children'),
    Input('option-selector', 'value')
)
def update_vol_sliders_container(selected_option):
    """
    Dynamically generates volatility sliders based on the selected option.

    Parameters:
    - selected_option (str or int): 'book' or the index of the selected option.

    Returns:
    - list of Dash components: The generated sliders.
    """
    if not book.basket:
        return html.Div("No Options in Book.")

    # We'll build a list of slider components
    slider_components = []

    if selected_option == 'book':
        # Show multiple sliders (one per option in the basket)
        for i, opt in enumerate(book.basket):
            if opt.use_vol_surface:
                # This option has a surface => "Surface Shift"
                slider_components.append(
                    html.Label(f"Option {i + 1} ({opt.type}) - Surface Shift")
                )
                slider_components.append(
                    dcc.Slider(
                        id={'type': 'vol-slider', 'index': i},  # pattern match
                        min=-0.1, max=0.1, step=0.01,
                        value=0.0,  # default shift
                        marks={round(v, 2): f"{round(v, 2)}" for v in np.linspace(-0.1, 0.1, 21)},
                        tooltip={"placement": "bottom", "always_visible": False}
                    )
                )
            else:
                # This option has a constant vol => "Constant Vol"
                slider_components.append(
                    html.Label(f"Option {i + 1} ({opt.type}) - Constant Vol")
                )
                slider_components.append(
                    dcc.Slider(
                        id={'type': 'vol-slider', 'index': i},
                        min=0.1, max=1.0, step=0.01,
                        value=opt.sigma or 0.2,  # if sigma is None, default to 0.2
                        marks={round(v, 1): f'{round(v, 1)}' for v in np.linspace(0.1, 1.0, 10)},
                        tooltip={"placement": "bottom", "always_visible": False}
                    )
                )
    else:
        # Single option chosen
        # Validate index
        if isinstance(selected_option, str):
            # fallback if something is off
            selected_option = 0
        opt_index = int(selected_option)
        if opt_index < 0 or opt_index >= len(book.basket):
            opt_index = 0

        opt = book.basket[opt_index]
        if opt.use_vol_surface:
            slider_components.append(
                html.Label(f"Option {opt_index + 1} ({opt.type}) - Surface Shift")
            )
            slider_components.append(
                dcc.Slider(
                    id={'type': 'vol-slider', 'index': opt_index},
                    min=-0.1, max=0.1, step=0.01,
                    value=0.0,
                    marks={round(v, 2): f"{round(v, 2)}" for v in np.linspace(-0.1, 0.1, 21)},
                    tooltip={"placement": "bottom", "always_visible": False}
                )
            )
        else:
            slider_components.append(
                html.Label(f"Option {opt_index + 1} ({opt.type}) - Constant Vol")
            )
            slider_components.append(
                dcc.Slider(
                    id={'type': 'vol-slider', 'index': opt_index},
                    min=0.1, max=1.0, step=0.01,
                    value=opt.sigma or 0.2,
                    marks={round(v, 1): f'{round(v, 1)}' for v in np.linspace(0.1, 1.0, 10)},
                    tooltip={"placement": "bottom", "always_visible": False}
                )
            )

    return slider_components

##########################################################
# 7) Positions Table Callback
##########################################################
@app.callback(
    Output('position-table', 'data'),
    Input('option-selector', 'value')
)
def update_position_table(selected_option):
    """
    Updates the positions table based on the selected option.

    Parameters:
    - selected_option (str or int): 'book' or the index of the selected option.

    Returns:
    - list of dict: The table data.
    """
    if not book.basket:
        return []
    df_positions = pd.DataFrame([
        {
            "Position": "Long" if opt.position > 0 else "Short",
            "Quantity": abs(opt.position),
            "Type": opt.type,
            "Strike": opt.K,
            "Maturity (days)": int(opt.T * 365),
            "Underlying Asset": opt.asset.name,
        }
        for opt in book.basket
    ])
    return df_positions.to_dict("records")

##########################################################
# 8) Main Callback: Apply Slider Values & Compute Greeks/PnL
##########################################################
@app.callback(
    [
        Output('delta-graph', 'figure'),
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
        Output('pnl-kpi-display', 'children'),
        Output('payoff-graph', 'figure')
    ],
    [
        Input('option-selector', 'value'),
        Input('price-slider', 'value'),
        Input({'type': 'vol-slider', 'index': ALL}, 'value')
    ]
)
def update_graphs(selected_option, price, vol_values_list):
    """
    Update all graphs and KPI displays based on selected option, underlying price, and vol sliders.

    Parameters:
    - selected_option (str or int): 'book' or the index of the selected option.
    - price (float): Current underlying price from the slider.
    - vol_values_list (list of float): List of volatility adjustments from sliders.

    Returns:
    - Tuple of plotly.graph_objects.Figure and Dash components: The updated figures and KPIs.
    """
    if not book.basket:
        empty_fig = go.Figure().update_layout(title="No data")
        empty_kpi = dbc.Row([dbc.Col(create_kpi_card("N/A", 0, "grey"), width=3)])
        return (empty_fig, empty_fig, empty_fig, empty_fig,
                empty_fig, empty_fig, empty_fig, empty_fig,
                empty_fig, empty_fig, empty_fig, empty_fig,
                empty_kpi, empty_kpi, empty_fig)

    # 1) Determine if 'book' or single option
    if selected_option == 'book':
        position_obj = book
        # We'll have len(vol_values_list) == len(book.basket),
        # one slider value per option in the order they appear.
    else:
        # single option
        position_index = 0
        if isinstance(selected_option, int):
            position_index = selected_option
        if position_index < 0 or position_index >= len(book.basket):
            position_index = 0
        position_obj = book.basket[position_index]
        # We'll have exactly 1 slider value in vol_values_list

    # 2) Update each relevant option's vol or shift
    #    Also set the underlying price.
    if selected_option == 'book':
        # Price updates for all
        if book.asset:
            book.asset.St = price
        else:
            mylogger.logger.warning("No underlying asset found in the book.")

        # Each slider value applies to the corresponding option in the basket
        for i, opt in enumerate(book.basket):
            if i >= len(vol_values_list):
                slider_val = 0.0
            else:
                slider_val = vol_values_list[i]
            if opt.use_vol_surface:
                shift_vol_surface(opt, slider_val)  # interpret as surface shift
            else:
                opt.sigma = slider_val  # interpret as constant vol
    else:
        # Single option scenario
        if position_obj.asset:
            position_obj.asset.St = price
        else:
            mylogger.logger.warning("No underlying asset found for the selected option.")

        if len(vol_values_list) > 0:
            slider_val = vol_values_list[0]
        else:
            slider_val = 0.0

        if position_obj.use_vol_surface:
            shift_vol_surface(position_obj, slider_val)
        else:
            position_obj.sigma = slider_val

    # 3) Compute Current Greeks for the selected position
    try:
        current_delta = position_obj.Delta_DF()
        current_gamma = position_obj.Gamma_DF()
        current_vega = position_obj.Vega_DF()
        current_theta = position_obj.Theta_DF()
        current_vanna = position_obj.Vanna_DF()
        current_volga = position_obj.Volga_DF()
        current_speed = position_obj.Speed_DF()
    except Exception as e:
        mylogger.logger.critical(f"Error computing Greeks: {e}")
        current_delta = current_gamma = current_vega = current_theta = 0
        current_vanna = current_volga = current_speed = 0

    # 4) Build Greek Figures
    delta_fig = plot_method(position_obj.DeltaRisk, price, current_delta, 'Delta')
    gamma_fig = plot_method(position_obj.GammaRisk, price, current_gamma, 'Gamma')
    vega_fig = plot_method(position_obj.VegaRisk, price, current_vega, 'Vega')
    theta_fig = plot_method(position_obj.ThetaRisk, price, current_theta, 'Theta')
    vanna_fig = plot_method(position_obj.VannaRisk, price, current_vanna, 'Vanna')
    volga_fig = plot_method(position_obj.VolgaRisk, price, current_volga, 'Volga')
    speed_fig = plot_method(position_obj.SpeedRisk, price, current_speed, 'Speed')

    # 5) PnL & Decomposition
    try:
        pnl_curve = position_obj.PnlRisk(plot=False)
        delta_pnl = safe_loc(position_obj.Delta_Pnl(plot=False), price)
        gamma_pnl = safe_loc(position_obj.Gamma_Pnl(plot=False), price)
        third_order_pnl = safe_loc(position_obj.Third_Order_Pnl(plot=False), price)
        n_order_pnl = safe_loc(position_obj.nOrderPnl(plot=False), price)
    except Exception as e:
        mylogger.logger.critical(f"Error computing PnL: {e}")
        pnl_curve = pd.Series()
        delta_pnl = gamma_pnl = third_order_pnl = n_order_pnl = 0

    pnl_fig = go.Figure(data=go.Scatter(
        x=pnl_curve.index, y=pnl_curve.values, mode='lines', name='PnL Curve'
    ))
    pnl_fig.update_layout(title='PnL Curve', xaxis_title='Underlying Price', yaxis_title='PnL')

    delta_pnl_fig = plot_method(position_obj.Delta_Pnl, price, delta_pnl, 'Delta PnL')
    gamma_pnl_fig = plot_method(position_obj.Gamma_Pnl, price, gamma_pnl, 'Gamma PnL')
    third_order_pnl_fig = plot_method(position_obj.Third_Order_Pnl, price, third_order_pnl, '3rd Order PnL')

    # 6) Payoff
    try:
        ST, payoffs = position_obj.display_payoff_option(plot=False)
        payoff_fig = go.Figure(data=go.Scatter(x=ST, y=payoffs, mode='lines', name='Payoff'))
        payoff_fig.update_layout(title='Payoff', xaxis_title='Underlying Price', yaxis_title='Payoff')

        current_payoff = position_obj.get_payoff_option(price)
        payoff_fig.add_trace(go.Scatter(
            x=[price], y=[current_payoff],
            mode='markers', marker=dict(color='red'), name='Current Payoff'
        ))
    except Exception as e:
        mylogger.logger.critical(f"Error plotting Payoff: {e}")
        payoff_fig = go.Figure().update_layout(title="Error plotting Payoff")

    # 7) KPI Cards
    greek_display = dbc.Row([
        dbc.Col(create_kpi_card("Delta", current_delta, "blue"), width=3),
        dbc.Col(create_kpi_card("Gamma", current_gamma, "orange"), width=3),
        dbc.Col(create_kpi_card("Vega", current_vega, "purple"), width=3),
        dbc.Col(create_kpi_card("Theta", current_theta, "red"), width=3),
        dbc.Col(create_kpi_card("Vanna", current_vanna, "green"), width=3),
        dbc.Col(create_kpi_card("Volga", current_volga, "teal"), width=3),
        dbc.Col(create_kpi_card("Speed", current_speed, "brown"), width=3)
    ], className='mb-3')

    pnl_display = dbc.Row([
        dbc.Col(create_kpi_card("Delta PnL", delta_pnl, "green"), width=3),
        dbc.Col(create_kpi_card("Gamma PnL", gamma_pnl, "tomato"), width=3),
        dbc.Col(create_kpi_card("3rd Order PnL", third_order_pnl, "orange"), width=3),
        dbc.Col(create_kpi_card("N-Order PnL", n_order_pnl, "goldenrod"), width=3)
    ], className='mb-3')

    return (
        delta_fig, gamma_fig, vega_fig, theta_fig,
        vanna_fig, volga_fig, speed_fig,
        pnl_fig, delta_pnl_fig, gamma_pnl_fig, third_order_pnl_fig,
        greek_display, pnl_display,
        payoff_fig
    )

##########################################################
# 9) Booking Callback: Handle New Trade Booking
##########################################################
@app.callback(
    Output('booking-feedback', 'children'),
    [
        Input('book-trade-button', 'n_clicks')
    ],
    [
        State('new-trade-position', 'value'),
        State('new-trade-type', 'value'),
        State('new-trade-asset', 'value'),
        State('new-trade-strike', 'value'),
        State('new-trade-maturity', 'value'),
        State('new-trade-quantity', 'value'),
        State('new-trade-booked-price', 'value'),
        State('new-trade-volatility', 'value'),
        State('tabs', 'value')
    ]
)
def book_new_trade(n_clicks, position, trade_type, asset_name, strike, maturity_days, quantity, booked_price, volatility, current_tab):
    """
    Handles the booking of a new trade and appends it to the booking Excel file.
    """

    # Validate inputs
    if not all([position, trade_type, asset_name, strike, maturity_days, quantity, booked_price, volatility]):
        return dbc.Alert("Please fill in all fields.", color="warning")

    # Additional Validation
    if strike <= 0:
        return dbc.Alert("Strike price must be positive.", color="danger")
    if volatility <= 0:
        return dbc.Alert("Volatility must be positive.", color="danger")
    if quantity <= 0:
        return dbc.Alert("Quantity must be a positive integer.", color="danger")

    asset = book.asset  # Assuming single asset as per existing code

    if not asset:
        return dbc.Alert("No underlying asset found in the book.", color="danger")

    # ASSET Booking
    if trade_type == "Asset":
        tobebooked = asset_BS(
            S0=booked_price,
            quantity=quantity if position == 'long' else -quantity,
            name=book.asset.name)
    else:
    # OPTION Booking
        option_type = trade_type
        # Calculate time to maturity in years
        try:
            maturity_days = int(maturity_days)
            if maturity_days <= 0:
                return dbc.Alert("Maturity must be a positive integer representing days.", color="danger")
            T = maturity_days / 365
        except ValueError:
            return dbc.Alert("Maturity must be an integer representing days.", color="danger")

        # Create a new Option instance
        try:
            tobebooked = Option_eu(
                position=quantity if position == 'long' else -quantity,
                type=option_type,
                asset=asset,
                K=strike,
                T=T,
                r=0.1,  # Adjust as needed or make it dynamic
                sigma=volatility,
                booked_price=booked_price
            )
        except Exception as e:
            mylogger.logger.critical(f"Error creating new option: {e}")
            return dbc.Alert(f"Error creating new option: {e}", color="danger")

    try:
        booking_request = Booking_Request(tobebooked)
        booking_request.run_Booking(lot_size=10000, book_name=default_book_name)
    except Exception as e:
        mylogger.logger.critical(f"Error executing booking request: {e}")
        return dbc.Alert(f"Error executing booking request: {e}", color="danger")

    # Provide feedback
    return dbc.Alert("Trade booked successfully!", color="success")

##########################################################
# Run the Dash App
##########################################################
if __name__ == '__main__':
    app.run_server(debug=True)
