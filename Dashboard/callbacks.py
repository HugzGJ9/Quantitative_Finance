# Dashboard/callbacks.py
import dash
import plotly.graph_objects as go
from dash import Input, Output, State, ALL, html
import dash_bootstrap_components as dbc
import pandas as pd
from Logger.Logger import mylogger
from Volatility.Volatility import SMILE
from .utils import (
    load_book, get_available_books, run_Mtm, shift_vol_surface, safe_loc,
    create_kpi_card, plot_method, asset_BS, Option_eu, Booking_Request
)

def register_callbacks(app):
    @app.callback(
        Output('empty-book-alert', 'children'),
        Input('current-book', 'data')
    )
    def handle_empty_book(current_book_name):
        if not current_book_name:
            return dbc.Alert("No book selected.", color="warning")
        book_obj = load_book(current_book_name, smile_df=SMILE)
        if not book_obj:
            return dbc.Alert("Book could not be loaded.", color="danger")
        if not book_obj.basket:
            # Show a dismissable alert with a suggestion to add a trade
            return dbc.Alert(
                [
                    html.Strong("This Book is currently empty. "),
                    "Why not book your first trade? Head to ",
                    html.A("Book New Trade", href="#",
                           style={"textDecoration": "underline", "color": "#0dcaf0"}),
                    " tab to add a position."
                ],
                color="info",
                dismissable=True
            )
        # If the book is not empty, show nothing
        return dash.no_update

    @app.callback(
        [
            Output('position-table', 'data'),
            Output('position-table', 'columns'),
            Output('mtm-feedback', 'children')
        ],
        [
            Input('run-mtm-button', 'n_clicks'),
            Input('current-book', 'data')
        ],
        [
            State('mtm-lot-size', 'value')
        ]
    )
    def execute_mtm(n_clicks, current_book_name, lot_size):
        """
        If book is empty, display a placeholder message instead of the usual MtM table.
        """
        if not current_book_name:
            return [], [], dbc.Alert("No book selected.", color="warning")

        # Try loading the book
        current_book = load_book(current_book_name, smile_df=SMILE)

        # If no book or empty book, return a placeholder
        if not current_book or not current_book.basket:
            # "Empty" placeholders
            columns = [{"name": "Book is empty", "id": "empty"}]
            data = [{"empty": "No data"}]
            feedback = dbc.Alert("Book is empty.", color="info")
            return data, columns, feedback

        # Book is not empty, so proceed with run_Mtm
        try:
            mtm_df = run_Mtm(LS=lot_size, book_name=current_book_name)

            # If run_Mtm returns None or not a DataFrame, handle gracefully
            if mtm_df is None or not isinstance(mtm_df, pd.DataFrame):
                feedback = dbc.Alert(
                    "No data returned by run_Mtm. Please check the book or data source.",
                    color="warning"
                )
                columns = [{"name": "Book is empty", "id": "no_data"}]
                data = [{"no_data": "Could not calculate MtM."}]
                return data, columns, feedback

            # Handle an empty DataFrame
            if mtm_df.empty:
                columns = [{"name": "Book is empty", "id": "no_positions"}]
                data = [{"no_positions": "No positions found."}]
                feedback = dbc.Alert(
                    "Your book has no positions. Consider booking a new trade.",
                    color="info"
                )
                return data, columns, feedback

            # Otherwise, proceed normally
            if mtm_df.shape[1] > 5:
                # Example of dropping last 5 columns if not needed
                mtm_df = mtm_df.iloc[:, :-5]

            float_cols = mtm_df.select_dtypes(include=['float']).columns
            mtm_df[float_cols] = mtm_df[float_cols].round(2)

            columns = [{"name": col, "id": col} for col in mtm_df.columns]
            data = mtm_df.to_dict('records')
            feedback = dbc.Alert("Mark-to-Market completed successfully.", color="success")
            return data, columns, feedback

        except Exception as e:
            mylogger.logger.error(f"Error during MtM process: {e}")
            feedback = dbc.Alert(f"Error during MtM process: {e}", color="danger")
            return [], [], feedback

    @app.callback(
        Output('vol-sliders-container', 'children'),
        Input('option-selector', 'value'),
        State('current-book', 'data')
    )
    def update_vol_sliders_container(selected_option, current_book_name):
        if not current_book_name:
            return dbc.Alert("No book selected.", color="warning")
        current_book = load_book(current_book_name, smile_df=SMILE)
        if not current_book or not current_book.basket:
            return dbc.Alert("No options in the book. Add a trade first.", color="info")

        sliders = []
        if selected_option == 'book':
            for i, opt in enumerate(current_book.basket):
                label = f"Option {i + 1} ({opt.type})"
                slider_id = {'type': 'vol-slider', 'index': i}
                if opt.use_vol_surface:
                    min_val, max_val, step, default, mark_step = (-0.1, 0.1, 0.01, 0.0, 0.02)
                else:
                    min_val, max_val, step, default, mark_step = (0.1, 1.0, 0.01, opt.sigma or 0.2, 0.1)
                marks = {round(v, 2): str(round(v, 2)) for v in
                    [min_val + j*mark_step for j in range(int((max_val-min_val)/mark_step)+1)]}
                sliders.append(dbc.Label(label, className='fw-bold mt-2'))
                sliders.append(dash.dcc.Slider(
                    id=slider_id,
                    min=min_val,
                    max=max_val,
                    step=step,
                    value=default,
                    marks=marks,
                    tooltip={"placement": "bottom", "always_visible": False}
                ))
        else:
            try:
                idx = int(selected_option)
                opt = current_book.basket[idx]
                label = f"Option {idx + 1} ({opt.type})"
                slider_id = {'type': 'vol-slider', 'index': idx}
                if opt.use_vol_surface:
                    min_val, max_val, step, default, mark_step = (-0.1, 0.1, 0.01, 0.0, 0.02)
                else:
                    min_val, max_val, step, default, mark_step = (0.1, 1.0, 0.01, opt.sigma or 0.2, 0.1)
                marks = {round(v, 2): str(round(v, 2)) for v in
                    [min_val + j*mark_step for j in range(int((max_val-min_val)/mark_step)+1)]}
                sliders.append(dbc.Label(label, className='fw-bold mt-2'))
                sliders.append(dash.dcc.Slider(
                    id=slider_id,
                    min=min_val,
                    max=max_val,
                    step=step,
                    value=default,
                    marks=marks,
                    tooltip={"placement": "bottom", "always_visible": False}
                ))
            except (IndexError, ValueError):
                mylogger.logger.warning("Selected option index out of range.")
                return dbc.Alert("Invalid option selected.", color="danger")
        return sliders

    @app.callback(
        Output('loaded-book-display', 'children'),
        Input('book-selector', 'value')
    )
    def update_loaded_book_display(selected_book):
        if selected_book:
            load_book(selected_book)
            return f"Loaded Book: {selected_book}"
        return "No Book Selected"

    @app.callback(
        Output('current-book', 'data'),
        Input('book-selector', 'value')
    )
    def update_current_book(selected_book):
        if selected_book:
            new_book = load_book(selected_book, smile_df=SMILE)
            if new_book is None:
                mylogger.logger.critical(f"Failed to load book: {selected_book}")
            return selected_book
        return None

    @app.callback(
        [
            Output('option-selector', 'options'),
            Output('option-selector', 'value'),
            Output('price-slider', 'min'),
            Output('price-slider', 'max'),
            Output('price-slider', 'step'),
            Output('price-slider', 'value'),
            Output('price-slider', 'marks'),
            Output('new-trade-asset', 'value')
        ],
        Input('current-book', 'data')
    )
    def update_option_selector_and_slider(current_book_name):
        if not current_book_name:
            return [], None, 80, 120, 0.1, 100, {}, ''
        current_book = load_book(current_book_name, smile_df=SMILE)
        if not current_book:
            return [], None, 80, 120, 0.1, 100, {}, ''
        option_options = [{'label': 'Book (All)', 'value': 'book'}]
        option_options += [{'label': f'Option {i + 1}: {opt.type}', 'value': i}
                           for i, opt in enumerate(current_book.basket)]
        selected_value = 'book' if option_options else None

        if current_book.asset:
            asset_price = current_book.asset.St
            min_price = round(0.8 * asset_price, 2)
            max_price = round(1.2 * asset_price, 2)
            initial_price = asset_price
        else:
            asset_price = 100
            min_price = 80
            max_price = 120
            initial_price = 100

        step = (max_price - min_price) / 100
        marks = {
            round(0.8 * asset_price, 2): str(round(0.8 * asset_price, 2)),
            round(0.9 * asset_price, 2): str(round(0.9 * asset_price, 2)),
            round(asset_price, 2): str(round(asset_price, 2)),
            round(1.1 * asset_price, 2): str(round(1.1 * asset_price, 2)),
            round(1.2 * asset_price, 2): str(round(1.2 * asset_price, 2))
        }
        asset_name = current_book.asset.name if current_book.asset else 'AAPL'
        return option_options, selected_value, min_price, max_price, step, initial_price, marks, asset_name

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
            Input({'type': 'vol-slider', 'index': ALL}, 'value'),
            Input('current-book', 'data')
        ]
    )
    def update_graphs(selected_option, price, vol_values_list, current_book_name):
        """
        If the book is empty, display "Book is empty" placeholders for figures/graphs/KPIs.
        """
        if not current_book_name:
            # Return placeholders (14 outputs in total)
            empty_fig = go.Figure().update_layout(title="Book is empty")
            empty_kpi = html.Div("Book is empty", style={"color": "lightgray", "textAlign": "center"})
            return [empty_fig] * 7 + [empty_fig] * 4 + [empty_kpi, empty_kpi, empty_fig]

        current_book = load_book(current_book_name, smile_df=SMILE)

        # Check if the book is empty
        if not current_book or not current_book.basket:
            empty_fig = go.Figure().update_layout(title="Book is empty")
            empty_kpi = html.Div("Book is empty", style={"color": "lightgray", "textAlign": "center"})
            return [empty_fig] * 7 + [empty_fig] * 4 + [empty_kpi, empty_kpi, empty_fig]

        # BOOK IS NOT EMPTY -> proceed normally
        # (Same logic as before: apply vol shifts, compute Greeks, build figures)
        # -----------------------------------------------------------------------
        # 1) Identify if user is looking at 'book' or an individual option
        if selected_option == 'book':
            selection = current_book
            vol_values = vol_values_list
        else:
            try:
                idx = int(selected_option)
                selection = current_book.basket[idx]
                vol_values = [vol_values_list[0]] if vol_values_list else [0.0]
            except (IndexError, ValueError):
                invalid_fig = go.Figure().update_layout(title="Invalid selection")
                empty_kpi = html.Div("Invalid Option", style={"color": "red", "textAlign": "center"})
                return [invalid_fig] * 7 + [invalid_fig] * 4 + [empty_kpi, empty_kpi, invalid_fig]

        if current_book.asset:
            current_book.asset.St = price

        # 2) Apply vol shifts
        if selected_option == 'book':
            for i, opt in enumerate(current_book.basket):
                if i < len(vol_values):
                    val = vol_values[i]
                    if opt.use_vol_surface:
                        shift_vol_surface(opt, val)
                    else:
                        opt.sigma = val
        else:
            if vol_values_list:
                val = vol_values_list[0]
                if selection.use_vol_surface:
                    shift_vol_surface(selection, val)
                else:
                    selection.sigma = val

        # 3) Compute Greeks (handle any errors)
        try:
            current_delta = selection.Delta_DF()
            current_gamma = selection.Gamma_DF()
            current_vega = selection.Vega_DF()
            current_theta = selection.Theta_DF()
            current_vanna = selection.Vanna_DF()
            current_volga = selection.Volga_DF()
            current_speed = selection.Speed_DF()

            # If any are Series/DataFrame, get the first row
            if isinstance(current_delta, (pd.Series, pd.DataFrame)): current_delta = current_delta.iloc[0]
            if isinstance(current_gamma, (pd.Series, pd.DataFrame)): current_gamma = current_gamma.iloc[0]
            if isinstance(current_vega, (pd.Series, pd.DataFrame)): current_vega = current_vega.iloc[0]
            if isinstance(current_theta, (pd.Series, pd.DataFrame)): current_theta = current_theta.iloc[0]
            if isinstance(current_vanna, (pd.Series, pd.DataFrame)): current_vanna = current_vanna.iloc[0]
            if isinstance(current_volga, (pd.Series, pd.DataFrame)): current_volga = current_volga.iloc[0]
            if isinstance(current_speed, (pd.Series, pd.DataFrame)): current_speed = current_speed.iloc[0]
        except Exception as e:
            mylogger.logger.critical(f"Error computing Greeks: {e}")
            current_delta = current_gamma = current_vega = current_theta = current_vanna = current_volga = current_speed = 0

        # 4) Build Figures
        delta_fig = plot_method(selection.DeltaRisk, price, current_delta, 'Delta')
        gamma_fig = plot_method(selection.GammaRisk, price, current_gamma, 'Gamma')
        vega_fig = plot_method(selection.VegaRisk, price, current_vega, 'Vega')
        theta_fig = plot_method(selection.ThetaRisk, price, current_theta, 'Theta')
        vanna_fig = plot_method(selection.VannaRisk, price, current_vanna, 'Vanna')
        volga_fig = plot_method(selection.VolgaRisk, price, current_volga, 'Volga')
        speed_fig = plot_method(selection.SpeedRisk, price, current_speed, 'Speed')

        # 5) PnL
        try:
            pnl_curve = selection.PnlRisk(plot=False)
            delta_pnl = safe_loc(selection.Delta_Pnl(plot=False), price)
            gamma_pnl = safe_loc(selection.Gamma_Pnl(plot=False), price)
            third_order_pnl = safe_loc(selection.Third_Order_Pnl(plot=False), price)
            n_order_pnl = safe_loc(selection.nOrderPnl(plot=False), price)
        except Exception as e:
            mylogger.logger.critical(f"Error computing PnL: {e}")
            pnl_curve = pd.Series()
            delta_pnl = 0
            gamma_pnl = 0
            third_order_pnl = 0
            n_order_pnl = 0

        pnl_fig = go.Figure(data=go.Scatter(x=pnl_curve.index, y=pnl_curve.values, mode='lines', name='PnL'))
        pnl_fig.update_layout(title='PnL Curve', xaxis_title='Underlying Price', yaxis_title='PnL',
                              template='plotly_dark')
        delta_pnl_fig = plot_method(selection.Delta_Pnl, price, delta_pnl, 'Delta PnL')
        gamma_pnl_fig = plot_method(selection.Gamma_Pnl, price, gamma_pnl, 'Gamma PnL')
        third_order_pnl_fig = plot_method(selection.Third_Order_Pnl, price, third_order_pnl, '3rd Order PnL')

        # 6) Payoff
        try:
            ST, payoffs = selection.display_payoff_option(plot=False)
            payoff_fig = go.Figure(data=go.Scatter(x=ST, y=payoffs, mode='lines', name='Payoff'))
            payoff_fig.update_layout(title='Payoff', xaxis_title='Underlying Price', yaxis_title='Payoff',
                                     template='plotly_dark')
            current_payoff = selection.get_payoff_option(price)
            payoff_fig.add_trace(go.Scatter(
                x=[price], y=[current_payoff],
                mode='markers', marker=dict(color='red'), name='Current Payoff'
            ))
        except Exception as e:
            mylogger.logger.critical(f"Error plotting Payoff: {e}")
            payoff_fig = go.Figure().update_layout(title="Error plotting Payoff")

        # 7) KPI displays
        greek_display = dbc.Row([
            dbc.Col(create_kpi_card("Delta", current_delta, "#2FA4E7"), width=3),
            dbc.Col(create_kpi_card("Gamma", current_gamma, "#FF851B"), width=3),
            dbc.Col(create_kpi_card("Vega", current_vega, "#B10DC9"), width=3),
            dbc.Col(create_kpi_card("Theta", current_theta, "#FF4136"), width=3),
            dbc.Col(create_kpi_card("Vanna", current_vanna, "#2ECC40"), width=3),
            dbc.Col(create_kpi_card("Volga", current_volga, "#39CCCC"), width=3),
            dbc.Col(create_kpi_card("Speed", current_speed, "#FFDC00"), width=3)
        ], className='mb-3')

        pnl_display = dbc.Row([
            dbc.Col(create_kpi_card("Delta PnL", delta_pnl, "#2ECC40"), width=3),
            dbc.Col(create_kpi_card("Gamma PnL", gamma_pnl, "#FF4136"), width=3),
            dbc.Col(create_kpi_card("3rd Order PnL", third_order_pnl, "#FF851B"), width=3),
            dbc.Col(create_kpi_card("N-Order PnL", n_order_pnl, "#FFD700"), width=3)
        ], className='mb-3')

        # Return the updated figures and KPI components
        return (
            delta_fig, gamma_fig, vega_fig, theta_fig, vanna_fig, volga_fig, speed_fig,
            pnl_fig, delta_pnl_fig, gamma_pnl_fig, third_order_pnl_fig,
            greek_display, pnl_display, payoff_fig
        )

    @app.callback(
        Output('booking-feedback', 'children'),
        [
            Input('book-trade-button', 'n_clicks')
        ],
        [
            State('new-trade-position', 'value'),
            State('new-trade-type', 'value'),
            State('new-trade-asset', 'value'),  # This is now from the dropdown
            State('new-trade-strike', 'value'),
            State('new-trade-maturity', 'value'),
            State('new-trade-quantity', 'value'),
            State('new-trade-booked-price', 'value'),
            State('new-trade-volatility', 'value'),
            State('current-book', 'data')
        ]
    )
    def book_new_trade(n_clicks, position, trade_type, asset_name, strike,
                       maturity_days, quantity, booked_price, volatility, current_book_name):
        if not current_book_name:
            return dbc.Alert("No book selected.", color="danger")
        current_book = load_book(current_book_name, smile_df=SMILE)
        if not current_book:
            return dbc.Alert("Book could not be loaded.", color="danger")
        if not all([position, trade_type, asset_name, strike, maturity_days, quantity, booked_price, volatility]):
            return dbc.Alert("Please fill in all fields.", color="warning")
        if strike <= 0:
            return dbc.Alert("Strike must be positive.", color="danger")
        if volatility <= 0:
            return dbc.Alert("Volatility must be positive.", color="danger")
        if quantity <= 0:
            return dbc.Alert("Quantity must be positive.", color="danger")

        if not current_book.basket:
            asset = asset_BS(100, quantity=0, name=asset_name)
        else:
            asset = current_book.asset
        if not asset:
            return dbc.Alert("No underlying asset found.", color="danger")

        if trade_type == "Asset":
            try:
                tobebooked = asset_BS(
                    S0=booked_price,
                    quantity=quantity if position == 'long' else -quantity,
                    name=asset_name
                )
            except Exception as e:
                mylogger.logger.critical(f"Error creating Asset position: {e}")
                return dbc.Alert(f"Error creating Asset: {e}", color="danger")
        else:
            try:
                maturity_days = int(maturity_days)
                if maturity_days <= 0:
                    return dbc.Alert("Maturity must be positive days.", color="danger")
                T = maturity_days / 365
            except ValueError:
                return dbc.Alert("Maturity must be an integer.", color="danger")

            try:
                tobebooked = Option_eu(
                    position=quantity if position == 'long' else -quantity,
                    type=trade_type,
                    asset=asset,
                    K=strike,
                    T=T,
                    r=0.1,
                    volatility_surface_df=SMILE,
                    use_vol_surface=True,
                    booked_price=booked_price
                )
            except Exception as e:
                mylogger.logger.critical(f"Error creating new option: {e}")
                return dbc.Alert(f"Error creating option: {e}", color="danger")

        try:
            booking_request = Booking_Request(tobebooked)
            booking_request.run_Booking(lot_size=10000, book_name=current_book_name)
        except Exception as e:
            mylogger.logger.critical(f"Error booking: {e}")
            return dbc.Alert(f"Error booking: {e}", color="danger")

        return dbc.Alert("Trade booked successfully!", color="success")
