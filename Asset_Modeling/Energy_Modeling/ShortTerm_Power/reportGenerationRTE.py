import pandas as pd
from API.RTE.data import getAPIdata
from API.SUPABASE.save import saveDailyGenerationTS
from Asset_Modeling.Energy_Modeling.ShortTerm_Power.buildGenerationReport import buildMonthlyTable, setImgEmail, style_html_table, table_style
from Asset_Modeling.Energy_Modeling.data.data import fetchRESGenerationMonthlyData
from Graphics.Graphics import DAauctionplot, ForecastGenplot
from API.GMAIL.auto_email_template import setAutoemail

df_forecast_gen = getAPIdata(APIname="Generation Forecast")
df_forecast_gen = df_forecast_gen.rename(columns={'SOLAR': 'SR'})

res_generation_month, res_generation_day = fetchRESGenerationMonthlyData("FR")
monthly_stats = buildMonthlyTable(res_generation_month, res_generation_day)
saveDailyGenerationTS(df_forecast_gen, 'RTE')

fig = ForecastGenplot(df_forecast_gen, show=False)
image_cid, img_data = setImgEmail(fig)

df_forecast_gen['datetime'] = pd.to_datetime(df_forecast_gen.index.astype(str).str[:10] + ' ' + df_forecast_gen.index.astype(str).str[11:19])
table_forecast_html = style_html_table(df_forecast_gen[['datetime', 'WIND', 'SR']].to_html(index=False, border=1))
table_forecast_total_html = style_html_table(df_forecast_gen[['SR', 'WIND']].resample('D').sum().to_html(index=False, border=1))


title = f'RTE RES Generation Forecast FR {df_forecast_gen["datetime"][:10].iloc[0]}'
body = f"""
<h2>Day-Ahead Generation Forecast Summary</h2>
<h3>DA Volume Forecast WIND & SOLAR</h3>
{table_forecast_total_html}

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