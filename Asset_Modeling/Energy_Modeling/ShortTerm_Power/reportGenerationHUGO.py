import pandas as pd

from API.SUPABASE.save import saveDailyGenerationTS
from Asset_Modeling.Energy_Modeling.ShortTerm_Power.buildGenerationReport import buildMonthlyTable, setImgEmail, \
    style_html_table, table_style
from Asset_Modeling.Energy_Modeling.data.data import fetchRESGenerationMonthlyData
from Graphics.Graphics import ForecastGenplot
from API.GMAIL.auto_email_template import setAutoemail
from io import BytesIO
from email.utils import make_msgid
from Model.Power.RESPowerGeneration_forecast import getGenerationForecastReport




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