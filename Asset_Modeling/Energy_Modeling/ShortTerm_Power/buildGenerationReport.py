from email.utils import make_msgid
import pandas as pd
from io import BytesIO
from API.GMAIL.auto_email_template import setAutoemail
from Asset_Modeling.Energy_Modeling.ShortTerm_Power.templates.generation import style_html_table
from Graphics.Graphics import ForecastGenplot


def buildMonthlyTable(res_generation, res_generation_day):
    now = pd.Timestamp.now()
    current_month = now.month
    current_year = now.year
    last_year = current_year - 1

    monthly_data = res_generation[res_generation.index.month == current_month]
    monthly_data_day = res_generation_day[res_generation_day.index.month == current_month]

    monthly_stats = pd.DataFrame({
        'Techno': monthly_data.columns,
        'q1 (25%) history': monthly_data.quantile(0.25),
        'mean history': monthly_data.mean(),
        'q3 (75%) history': monthly_data.quantile(0.75),
        'mean_last_year': res_generation[
            (res_generation.index.month == current_month) &
            (res_generation.index.year == last_year)
            ].mean(),
        'mean_current_year': monthly_data_day[monthly_data_day.index.year == current_year].mean(),
        'q1_current_year (25%)': monthly_data_day[monthly_data_day.index.year == current_year].quantile(0.25),
        'q3_current_year (75%)': monthly_data_day[monthly_data_day.index.year == current_year].quantile(0.75),
    }).reset_index(drop=True)

    return monthly_stats


def setImgEmail(fig):
    img_data = BytesIO()
    fig.savefig(img_data, format='png')
    img_data.seek(0)
    image_cid = make_msgid(domain='xyz.com')[1:-1]
    return image_cid, img_data


def buildGenerationForecastEmail(title, tables, img_data, image_cid):
    table_forecast_total_html, table_monthly_stats_html, table_forecast_html = tables

    body = f"""
    <h2 style="color:#2F4F4F;">Day-Ahead Generation Forecast Summary</h2>

    <h3>⚡ Total Forecast by Day (WIND & SOLAR)</h3>
    {table_forecast_total_html}

    <h3>📊 Monthly Stats</h3>
    {table_monthly_stats_html}

    <h3>📈 Forecast Chart</h3>
    <img src="cid:{image_cid}" style="max-width:100%; height:auto;">

    <h3>📅 Hourly Forecast (Prices & Volumes)</h3>
    {table_forecast_html}
    """

    recipients = ['hugo.lambert.perso@gmail.com', 'hugo.lambert.perso@gmail.com']
    setAutoemail(recipients, title, body, image_buffers=[img_data], image_cids=[image_cid])


def buildAuctionDAEmail(title, tables, img_data, image_cid):
    table_price_summary_html, table_price_html = tables

    body = f"""
    <h2 style="color:#2F4F4F;">Day-Ahead Auction Summary</h2>

    <h3>💶 Price & Volume Summary</h3>
    {table_price_summary_html}

    <h3>📉 Price Curve & Volume Histogram</h3>
    <img src="cid:{image_cid}" style="max-width:100%; height:auto;">

    <h3>📅 Hourly Market Data</h3>
    {table_price_html}
    """

    recipients = ['hugo.lambert.perso@gmail.com', 'hugo.lambert.perso@gmail.com']
    setAutoemail(recipients, title, body, image_buffers=[img_data], image_cids=[image_cid])


def fetchGenerationReport(generation_forecast, monthly_stats):
    # Normalize to MW and round
    generation_forecast = (generation_forecast / 1000).round(2)
    monthly_stats.iloc[:, 1:] = (monthly_stats.iloc[:, 1:] / 1000).round(2)

    # Generate and embed figure
    fig = ForecastGenplot(generation_forecast, show=False)
    image_cid, img_data = setImgEmail(fig)

    # Format datetime for HTML
    generation_forecast['datetime'] = pd.to_datetime(
        generation_forecast.index.strftime('%Y-%m-%d %H:%M:%S'))

    # Prepare tables
    forecast_columns = ['datetime', 'WIND', 'SR']
    table_forecast_html = style_html_table(
        generation_forecast[forecast_columns].to_html(index=False, border=1))

    table_forecast_total_html = style_html_table(
        generation_forecast[['SR', 'WIND']].resample('D').sum().to_html(index=False, border=1))

    table_monthly_stats_html = style_html_table(monthly_stats.to_html(index=False, border=1))

    return [table_forecast_total_html, table_monthly_stats_html, table_forecast_html], image_cid, img_data
