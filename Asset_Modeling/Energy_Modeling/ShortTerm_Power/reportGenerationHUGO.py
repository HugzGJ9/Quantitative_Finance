import pandas as pd

from API.SUPABASE.save import saveDailyGenerationTS
from Asset_Modeling.Energy_Modeling.data.data import fetchRESGenerationMonthlyData
from Graphics.Graphics import ForecastGenplot
from API.GMAIL.auto_email_template import setAutoemail
from io import BytesIO
from email.utils import make_msgid
from Model.Power.RESPowerGeneration_forecast import getGenerationForecastReport

def buildMonthlyTable(res_generation, res_generation_day):
    monthly_mean = res_generation[res_generation.index.month == pd.Timestamp.now().month].mean()
    monthly_q1 = res_generation[res_generation.index.month == pd.Timestamp.now().month].quantile(0.25)
    monthly_q3 = res_generation[res_generation.index.month == pd.Timestamp.now().month].quantile(0.75)

    monthly_mean_last_year = res_generation[
        (res_generation.index.month == pd.Timestamp.now().month) &
        (res_generation.index.year == (pd.Timestamp.now().year - 1))
        ].mean()
    monthly_mean_current_year = res_generation_day[
        (res_generation_day.index.month == pd.Timestamp.now().month) &
        (res_generation_day.index.year == (pd.Timestamp.now().year))
        ].mean()
    monthly_q1_current_year = res_generation_day[
        (res_generation_day.index.month == pd.Timestamp.now().month) &
        (res_generation_day.index.year == (pd.Timestamp.now().year))
        ].quantile(0.25)
    monthly_q3_current_year = res_generation_day[
        (res_generation_day.index.month == pd.Timestamp.now().month) &
        (res_generation_day.index.year == (pd.Timestamp.now().year))
        ].quantile(0.75)

    monthly_stats = pd.DataFrame({
        'Techno': monthly_q1.index,
        'q1 (25%) history': monthly_q1,
        'mean history': monthly_mean,
        'q3 (75%) history': monthly_q3,
        'mean_last_year': monthly_mean_last_year,
        'mean_current_year': monthly_mean_current_year,
        'q1_current_year (25%)': monthly_q1_current_year,
        'q3_current_year (75%)': monthly_q3_current_year})

    return monthly_stats
def setImgEmail(fig):
    img_data = BytesIO()
    fig.savefig(img_data, format='png')
    img_data.seek(0)
    image_cid = make_msgid(domain='xyz.com')[1:-1]
    return image_cid, img_data


table_style = """
<style>
    .summary-table {
        border-collapse: collapse;
        width: 100%;
        margin-bottom: 20px;
        font-family: Arial, sans-serif;
        font-size: 14px;
    }
    .summary-table th {
        background-color: #f2f2f2;
        color: #333;
        text-align: center;
        padding: 8px;
        border: 1px solid #ddd;
    }
    .summary-table td {
        padding: 8px;
        border: 1px solid #ddd;
        text-align: center;
    }
    .summary-table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
</style>
"""

generation_forecast = getGenerationForecastReport()
res_generation_month, res_generation_day = fetchRESGenerationMonthlyData("FR")
monthly_stats = buildMonthlyTable(res_generation_month, res_generation_day)
saveDailyGenerationTS(generation_forecast, 'HUGO')
# Convert MWh to GWh and round to 2 decimal places
generation_forecast = generation_forecast / 1000
generation_forecast = generation_forecast.round(2)
for col in monthly_stats.columns[1:]:
    monthly_stats[col] = (monthly_stats[col] / 1000).round(2)


fig = ForecastGenplot(generation_forecast, show=False)
image_cid, img_data = setImgEmail(fig)
def style_html_table(html):
    return html.replace(
        "<table border=\"1\" class=\"dataframe\">",
        """<table style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 14px;">"""
    ).replace(
        "<th>",
        """<th style="background-color: #f2f2f2; color: #333; text-align: center; padding: 8px; border: 1px solid #ddd;">"""
    ).replace(
        "<td>",
        """<td style="padding: 8px; border: 1px solid #ddd; text-align: center;">"""
    )

generation_forecast['datetime'] = pd.to_datetime(generation_forecast.index.astype(str).str[:10] + ' ' + generation_forecast.index.astype(str).str[11:19])
table_forecast_html = style_html_table(generation_forecast[['datetime', 'WIND', 'SR']].to_html(index=False, border=1))
table_forecast_total_html = style_html_table(generation_forecast[['SR', 'WIND']].resample('D').sum().to_html(index=False, border=1))
table_monthly_stats_html = style_html_table(monthly_stats.to_html(index=False, border=1))

title = f'Hugo RES Generation Forecast FR {generation_forecast["datetime"][:10].iloc[0]}'
body = f"""
<h2>Day-Ahead Generation Forecast Summary</h2>
<h3>DA Volume Forecast WIND & SOLAR</h3>
{table_forecast_total_html}
{table_monthly_stats_html}

<h3>Below is the price curve and volume histogram:</h3>
<img src="cid:{image_cid}">

<h3>Day-Ahead Prices and Volumes</h3>
{table_forecast_html}

"""
# Send email with image embedded
setAutoemail(
    ['hugo.lambert.perso@gmail.com', 'hugo.lambert.perso@gmail.com'],
    title,
    body,
    image_buffers=[img_data],
    image_cids=[image_cid]
)