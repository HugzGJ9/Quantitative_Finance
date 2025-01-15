# Dashboard/utils.py
import os, glob
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import html
from Logger.Logger import mylogger
from Booking.Option_Book_Management import importBook
from Booking.Run_Mtm import run_Mtm
from Volatility.Volatility import SMILE
from Asset_Modeling.Asset_class import asset_BS
from Options.Options_class import Option_eu
from Booking.Booking_Request import Booking_Request

BOOK_FILES_PATH = os.path.join('Booking', 'Book_Files')
cache_duration = datetime.timedelta(minutes=5)
last_fetch_time = None
cached_price = None

def get_available_books():
    pattern = os.path.join(BOOK_FILES_PATH, '*.xlsx')
    files = glob.glob(pattern)
    return [os.path.splitext(os.path.basename(file))[0] for file in files]

def load_book(book_name=None, smile_df=None):
    try:
        run_Mtm(LS=10000, book_name=book_name)
        book = importBook(book_name=book_name)
        if book is None:
            mylogger.logger.critical("importBook returned None.")
            return None
        mylogger.logger.info(f"Book loaded: {book_name}")
        return book
    except Exception as e:
        mylogger.logger.critical(f"Error loading book: {e}")
        return None

def shift_vol_surface(opt_obj, shift_value):
    if not opt_obj.use_vol_surface or opt_obj.volatility_surface_df is None:
        return
    vol_cols = opt_obj.volatility_surface_df.columns[1:]
    opt_obj.volatility_surface_df[vol_cols] += shift_value
    opt_obj.set_volatility_surface()

def safe_loc(series_or_df, key):
    if isinstance(series_or_df, (pd.Series, pd.DataFrame)) and key in series_or_df.index:
        return series_or_df.loc[key]
    return 0

def create_kpi_card(title, value, color):
    try:
        formatted_value = f"{float(value):.2f}"
    except (ValueError, TypeError):
        formatted_value = "N/A"
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className='card-title', style={'color': color}),
            html.H3(formatted_value, className='card-text')
        ]),
        className="shadow-sm text-center mb-2",
        style={'border-radius': '10px', 'backgroundColor': '#2b2b2b'}
    )

def plot_method(method, price, current_value, title):
    try:
        curve = method(plot=False)
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
        fig.update_layout(
            title=f'{title} Curve',
            xaxis_title='Underlying Price',
            yaxis_title=title,
            template='plotly_dark'
        )
        return fig
    except Exception as e:
        mylogger.logger.critical(f"Error plotting {title}: {e}")
        return go.Figure().update_layout(title=f"Error plotting {title}")
