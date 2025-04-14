import pandas as pd
from API.RTE.data import getAPIdata
from Graphics.Graphics import DAauctionplot

from API.GMAIL.auto_email_template import setAutoemail

from io import BytesIO
from email.utils import make_msgid

df = getAPIdata(APIname="Wholesale Market")
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
df = df.sort_values('datetime')

fig = DAauctionplot(df, title=f'Prix et Volume Power FR - {df["date"].iloc[0]}', show=False)

# Save figure to buffer
img_data = BytesIO()
fig.savefig(img_data, format='png')
img_data.seek(0)
image_cid = make_msgid(domain='xyz.com')[1:-1]

# Email content
title = f'DA auction FR {df["date"].iloc[0]}'
body = f"""
<h2>Day-Ahead Auction Summary</h2>
<p>Below is the price curve and volume histogram:</p>
<img src="cid:{image_cid}">
"""

# Send email with image embedded
setAutoemail(
    ['hugo.lambert.perso@gmail.com', 'hugo.lambert.perso@gmail.com'],
    title,
    body,
    image_buffer=img_data,
    image_cid=image_cid
)