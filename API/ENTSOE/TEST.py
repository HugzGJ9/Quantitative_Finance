from entsoe import EntsoePandasClient

client = EntsoePandasClient(api_key='e37c0206-5d3c-4258-84d6-323ac6044d69')

import pandas as pd

start = pd.Timestamp('20250201T0001', tz='Europe/Paris')
end = pd.Timestamp('20250413T2359', tz='Europe/Paris')

df = client.query_generation(
    country_code='FR',
    start=start
)
df['hour'] = df.index.hour
df.groupby(['hour']).sum()
df