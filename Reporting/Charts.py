import datetime
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def build_correl_matrix(df):
    # Exclude non-numeric columns before computing correlation
    numeric_df = df.select_dtypes(include=[np.number])

    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()

    # Plot correlation matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(corr_matrix, cmap='coolwarm')

    fig.colorbar(cax)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45)
    ax.set_yticklabels(corr_matrix.columns)

    # Annotate correlation values
    for (i, j), val in np.ndenumerate(corr_matrix):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                color='white' if abs(val) > 0.5 else 'black')

    plt.title('Correlation Matrix', pad=20)
    plt.tight_layout()
    plt.show()
def create_plotly_charts(df, df_1h, df_15m, vol_weekly, vol_daily, vol_intraday_hourly):
    # Style configuration
    template = "plotly_white"
    color_palette = {
        'positive': '#2ecc71',
        'negative': '#e74c3c',
        'primary': '#3498db',
        'neutral': '#95a5a6'
    }

    # Candlestick Chart with annotations
    fig_candlestick = go.Figure(data=[go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'],
        close=df['Close'], name='Price',
        increasing_line_color=color_palette['positive'],
        decreasing_line_color=color_palette['negative'])])
    fig_candlestick.update_layout(
        title="Price Analysis with Volume Profile",
        xaxis_title="Date",
        yaxis_title="Price",
        template=template,
        height=500,
        xaxis_rangeslider_visible=False,
        xaxis=dict(tickformat='%Y-%m-%d'),
    )
    threshold_date = pd.to_datetime(df_15m['Date'].iloc[-1]).normalize()  # strips time
    df_15minID = df_15m[df_15m['Date'] > (threshold_date - pd.Timedelta(days=1))]
    fig_candlestickID = go.Figure(data=[go.Candlestick(
        x=df_15minID['Date'].dt.strftime('%d %H:%M'), open=df_15minID['Open'], high=df_15minID['High'], low=df_15minID['Low'],
        close=df_15minID['Close'], name='Price',
        increasing_line_color=color_palette['positive'],
        decreasing_line_color=color_palette['negative'])])
    fig_candlestickID.update_layout(
        title="Price Analysis with Volume Profile",
        xaxis_title="Date",
        yaxis_title="Price",
        template=template,
        height=500,
        xaxis_rangeslider_visible=False,
        xaxis=dict(tickformat='%Y-%m-%d'),
    )
    # Price and Volume Chart with range selector
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'], mode='lines',
        name='Close Price', line=dict(color=color_palette['primary'])))
    fig_price.add_trace(go.Bar(
        x=df.index, y=df['Volume'], name='Volume',
        marker_color=color_palette['neutral'], opacity=0.5,
        yaxis='y2'))
    fig_price.update_layout(
        title="Price and Volume Analysis",
        xaxis=dict(
            title="Date",
            tickformat='%Y-%m-%d',
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1W", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            )
        ),
        yaxis=dict(title="Price", side="left", showgrid=False),
        yaxis2=dict(title="Volume", side="right", overlaying="y", showgrid=False),
        template=template,
        height=400
    )

    # Enhanced RSI Chart
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=df['Date'], y=df['RSI'], mode='lines',
        name='RSI', line=dict(color=color_palette['primary'])))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color=color_palette['negative'])
    fig_rsi.add_hline(y=30, line_dash="dash", line_color=color_palette['positive'])
    fig_rsi.update_layout(
        title="RSI (14-period)",
        xaxis_title="Date",
        yaxis_title="RSI Value",
        xaxis=dict(tickformat='%Y-%m-%d'),
        template=template,
        height=300,
        shapes=[
            dict(type="rect", y0=70, y1=100, x0=0, x1=1,
                 xref="paper", fillcolor="rgba(231, 76, 60, 0.1)", line_width=0),
            dict(type="rect", y0=0, y1=30, x0=0, x1=1,
                 xref="paper", fillcolor="rgba(46, 204, 113, 0.1)", line_width=0)
        ]
    )

    fig_rsiID = go.Figure()
    fig_rsiID.add_trace(go.Scatter(
        x=df_15minID['Date'], y=df_15minID['RSI'], mode='lines',
        name='RSI', line=dict(color=color_palette['primary'])))
    fig_rsiID.add_hline(y=70, line_dash="dash", line_color=color_palette['negative'])
    fig_rsiID.add_hline(y=30, line_dash="dash", line_color=color_palette['positive'])
    fig_rsiID.update_layout(
        title="RSI ID (14-period)",
        xaxis_title="Date",
        yaxis_title="RSI Value",
        xaxis=dict(tickformat='%Y-%m-%d'),
        template=template,
        height=300,
        shapes=[
            dict(type="rect", y0=70, y1=100, x0=0, x1=1,
                 xref="paper", fillcolor="rgba(231, 76, 60, 0.1)", line_width=0),
            dict(type="rect", y0=0, y1=30, x0=0, x1=1,
                 xref="paper", fillcolor="rgba(46, 204, 113, 0.1)", line_width=0)
        ]
    )

    # MACD Chart with colored histogram
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(
        x=df['Date'], y=df['MACD'], mode='lines',
        name='MACD', line=dict(color=color_palette['primary'])))
    fig_macd.add_trace(go.Scatter(
        x=df['Date'], y=df['MACD_Signal'], mode='lines',
        name='Signal', line=dict(color=color_palette['neutral'], dash='dot')))
    colors = [color_palette['positive'] if val >= 0 else color_palette['negative'] for val in df['MACD_Hist']]
    fig_macd.add_trace(go.Bar(
        x=df['Date'],
        y=df['MACD_Hist'],
        marker_color=colors,
        name='MACD Histogram'
    ))
    fig_macd.update_layout(
        title="MACD Analysis",
        xaxis_title="Date",
        xaxis=dict(tickformat='%Y-%m-%d'),
        yaxis_title="Value",
        template=template,
        height=400,
        hovermode="x unified"
    )

    # Volatility Charts
    fig_vol_weekly = go.Figure()
    if vol_weekly:
        dates_weekly = [datetime.datetime.now() - datetime.timedelta(weeks=len(vol_weekly)-i) for i in range(len(vol_weekly))]
        fig_vol_weekly.add_trace(go.Scatter(
            x=dates_weekly,
            y=[v * 100 for v in vol_weekly],
            mode='lines+markers',
            name='Weekly Volatility',
            line=dict(width=2, color=color_palette['primary']),
            marker=dict(size=6, color=color_palette['primary']),
            fill='tozeroy'
        ))
    fig_vol_weekly.update_layout(
        title="Weekly Volatility Trend",
        xaxis_title="Date",
        xaxis=dict(tickformat='%Y-%m-%d'),
        yaxis_title="Volatility (%)",
        template=template
    )

    fig_vol_daily = go.Figure()
    if vol_daily:
        dates_daily = [datetime.datetime.now() - datetime.timedelta(days=len(vol_daily)-i) for i in range(len(vol_daily))]
        fig_vol_daily.add_trace(go.Scatter(
            x=dates_daily,
            y=[v * 100 for v in vol_daily],
            name='Daily Volatility',
            line=dict(width=2, color=color_palette['primary']),
            marker=dict(size=6, color=color_palette['primary']),
            fill='tozeroy'
        ))
    fig_vol_daily.update_layout(
        title="Daily Volatility Trend",
        xaxis_title="Date",
        xaxis=dict(tickformat='%Y-%m-%d'),
        yaxis_title="Volatility (%)",
        template=template
    )

    fig_vol_intraday = go.Figure()
    if vol_intraday_hourly:
        dates_intraday = [datetime.datetime.now() - datetime.timedelta(hours=len(vol_intraday_hourly)-i) for i in range(len(vol_intraday_hourly))]
        fig_vol_intraday.add_trace(go.Scatter(
            x=dates_intraday,
            y=[v * 100 for v in vol_intraday_hourly],
            name='Intraday Volatility',
            line=dict(width=2, color=color_palette['primary']),
            marker=dict(size=6, color=color_palette['primary']),
            fill='tozeroy'
        ))
    fig_vol_intraday.update_layout(
        title="Intraday Volatility Trend",
        xaxis_title="Date",
        xaxis=dict(tickformat='%Y-%m-%d %H:%M'),
        yaxis_title="Volatility (%)",
        template=template
    )
    correl_matrix_list = []
    for dataframe in [df, df_1h, df_15m]:
        correl_matrix_list.append(build_correl_matrix(dataframe))

    return (
        fig_candlestick, fig_price, fig_rsi, fig_macd,
        fig_vol_weekly, fig_vol_daily, fig_vol_intraday, fig_candlestickID, fig_rsiID
    )
