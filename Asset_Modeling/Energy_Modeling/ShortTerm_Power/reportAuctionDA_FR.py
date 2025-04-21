import pandas as pd
from API.RTE.data import getAPIdata
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

df = getAPIdata(APIname="Wholesale Market")
df_forecast_gen = getAPIdata(APIname="Generation Forecast")

fig = DAauctionplot(df, title=f'Prix et Volume Power FR - {df["date"].iloc[0]}', show=False)
fig2 = ForecastGenplot(df_forecast_gen, show=False)
image_cid, img_data = setImgEmail(fig)
image_cid2, img_data2 = setImgEmail(fig2)
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
df.index = df['datetime']
# df_forecast_gen['datetime'] = pd.to_datetime(df_forecast_gen.index.astype(str).str[:10] + ' ' + df_forecast_gen.index.astype(str).str[11:19])
table_price_html = df[['datetime', 'value', 'price']]
# table_forecast_html = style_html_table(df_forecast_gen[['datetime', 'WIND', 'SOLAR']].to_html(index=False, border=1))
# table_forecast_total_html = style_html_table(df_forecast_gen[['SOLAR', 'WIND']].resample('D').sum().to_html(index=False, border=1))

peak_mask = df['datetime'].dt.hour.between(8, 19, inclusive='both')
offpeak_mask = ~peak_mask

summary_df = pd.concat({
    'Base': df[['value', 'price']].mean(),
    'Peak': df.loc[peak_mask,    ['value', 'price']].mean(),
    'Off‑peak': df.loc[offpeak_mask, ['value', 'price']].mean()
}, axis=1).T
summary_df['Product'] = summary_df.index
summary_df = summary_df.rename(columns={'value': 'Volume',
                                        'price': 'Price'})
summary_df = summary_df[['Product', 'Volume', 'Price']]
table_price_html = table_price_html.rename(columns={'datetime': 'Datetime',
                                        'value': 'Volume',
                                        'price': 'Price'})

table_price_summary_html = style_html_table(summary_df.to_html(index=False, border=1))
table_price_html = style_html_table(table_price_html.to_html(index=False, border=1))

title = f'DA auction FR {df["date"].iloc[0]}'
body = f"""
<h2>Day-Ahead Auction Summary</h2>
<h3>DA Price & Volume</h3>
{table_price_summary_html}

<h3>Below is the price curve and volume histogram:</h3>
<img src="cid:{image_cid}">

<h3>Day-Ahead Prices and Volumes</h3>
{table_price_html}

"""
# Send email with image embedded
setAutoemail(
    ['hugo.lambert.perso@gmail.com', 'hugo.lambert.perso@gmail.com'],
    title,
    body,
    image_buffers=[img_data],
    image_cids=[image_cid]
)