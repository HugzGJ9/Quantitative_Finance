import pandas as pd
from API.RTE.data import getAPIdata
from API.SUPABASE.save import saveDailyGenerationTS
from Asset_Modeling.Energy_Modeling.ShortTerm_Power.reportGenerationHUGO import buildMonthlyTable
from Asset_Modeling.Energy_Modeling.data.data import fetchRESGenerationMonthlyData
from Graphics.Graphics import DAauctionplot, ForecastGenplot
from API.GMAIL.auto_email_template import setAutoemail
from io import BytesIO
from email.utils import make_msgid

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
df_forecast_gen = getAPIdata(APIname="Generation Forecast")
df_forecast_gen = df_forecast_gen.rename(columns={'SOLAR': 'SR'})

res_generation_month, res_generation_day = fetchRESGenerationMonthlyData("FR")
monthly_stats = buildMonthlyTable(res_generation_month, res_generation_day)
saveDailyGenerationTS(df_forecast_gen, 'RTE')

fig = ForecastGenplot(df_forecast_gen, show=False)
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

# Create raw HTML from pandas
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