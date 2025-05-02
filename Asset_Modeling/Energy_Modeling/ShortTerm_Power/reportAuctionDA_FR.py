import pandas as pd
from API.RTE.data import getAPIdata
from Asset_Modeling.Energy_Modeling.ShortTerm_Power.buildGenerationReport import buildAuctionDAEmail, setImgEmail
from Asset_Modeling.Energy_Modeling.ShortTerm_Power.templates.generation import style_html_table
from Graphics.Graphics import DAauctionplot, ForecastGenplot



df = getAPIdata(APIname="Wholesale Market")
df_forecast_gen = getAPIdata(APIname="Generation Forecast")
df_forecast_gen = df_forecast_gen.rename(columns={'SOLAR': 'SR'})

fig = DAauctionplot(df, title=f'Prix et Volume Power FR - {df["date"].iloc[0]}', show=False)
fig2 = ForecastGenplot(df_forecast_gen, show=False)
image_cid, img_data = setImgEmail(fig)
image_cid2, img_data2 = setImgEmail(fig2)

df.index = df['datetime']
table_price_html = df[['datetime', 'value', 'price']]
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
tables = [table_price_summary_html, table_price_html]
title = f'DA auction FR {df["date"].iloc[0]}'
buildAuctionDAEmail(title=title, tables=tables, image_cid=image_cid, img_data=img_data)