import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from API.SUPABASE.client import getDfSupabase, getAccessSupabase, getRowsSupabase
from Asset_Modeling.Energy_Modeling.PPA.ComputeRES_shape import AVG_price, VWA_price
from Asset_Modeling.Energy_Modeling.PPA.stats import buildSyntheticGeneration
from Logger.Logger import mylogger
from pytz.exceptions import AmbiguousTimeError, NonExistentTimeError
def datetime_generation(start_date: str, end_date: str) -> pd.DatetimeIndex:
    full_range = pd.date_range(start=start_date, end=end_date, freq="H", inclusive="left")
    return pd.DatetimeIndex(full_range)

class PPA():
    def __init__(self, site_name="Montlucon", techno="SR", capacity=10.0, pricing_type='fixed_price', start_date='01/01/2026', end_date='01/01/2027', id=None):
        if not id:
            self.id = random.randint(1, 10000000)
            self.site_name = site_name
            self.start_date = start_date
            self.end_date = end_date
            self.capacity = capacity
            self.techno = techno
            self.pricing_type = pricing_type
            self.country = "FR" if self.site_name in ['Montlucon'] else None
            self.proxy = pd.DataFrame()
            self.mark = pd.DataFrame()
            self.p50 = None
        else:

            row = getRowsSupabase('PPA', [id])
            self.id = row.iloc[0]['id']
            self.site_name = row.iloc[0]['site_name']
            self.start_date = row.iloc[0]['start_date']
            self.end_date = row.iloc[0]['end_date']
            self.capacity = row.iloc[0]['capacity']
            self.techno = row.iloc[0]['techno']
            self.pricing_type = row.iloc[0]['pricing_type']
            self.country = row.iloc[0]['country']
            self.proxy = pd.read_json(row.iloc[0]['proxy'])
            self.mark = pd.read_json(row.iloc[0]['mark'])
            self.p50 = row.iloc[0]['p50']

    def buildProxy(self):
        """
        Build an hourly solar-generation proxy whose mean for *each*
        season-hour combination matches SR_LoadFactor_FR.
        Stores result in `self.proxy` (column 'generation_mw') and returns it.
        """
        if not self.proxy.empty:
            mylogger.logger.warning('Proxy TS already set.')
            return
        else:
            if self.country != "FR":
                raise NotImplementedError("FR only")

            import pandas as pd
            from Profile_modelisation import SR_LoadFactor_FR2

            weather = getDfSupabase("WeatherFR")
            weather["id"] = pd.to_datetime(weather["id"], utc=True)

            weather = (weather
                       .set_index("id")
                       .sort_index()
                       .loc[:, ["Solar_Radiation"]])

            commissioning = pd.to_datetime(self.start_date, dayfirst=True, utc=True)
            weather = weather.loc[: commissioning - pd.Timedelta(seconds=1)]

            weather.index = weather.index.tz_convert('Europe/Paris')
            weather['hour'] = weather.index.hour
            for month in list(range(1, 13, 1)):
                mylogger.logger.info(f'{month=}')
                for hour in list(range(0, 24, 1)):
                    mylogger.logger.info(f'{hour=}')

                    df = weather[(weather['hour'] == hour) & (weather.index.month == month)]
                    mean_target = SR_LoadFactor_FR2[month][SR_LoadFactor_FR2.index == hour].values[0] * self.capacity

                    gen_temp = buildSyntheticGeneration(df, mean_target, capacity=self.capacity, sat_threshold=850)
                    self.proxy = pd.concat([self.proxy, gen_temp[['generation']]], ignore_index=False)
            self.proxy = self.proxy.sort_index()
            return self.proxy

    def saveInstance(self):
        supabase = getAccessSupabase('PPA')
        instance = pd.DataFrame([{
            'id': self.id,
            'site_name': self.site_name,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'capacity': self.capacity,
            'techno': self.techno,
            'pricing_type': self.pricing_type,
            'country': self.country,
            'proxy': self.proxy.to_json(),
            'mark': self.mark.to_json(),
            'p50': self.p50
        }])

        instance.set_index('id')
        # supabase.table('PPA').delete().in_('id', instance.index).execute()

        for col in instance.columns:
            if pd.api.types.is_datetime64_any_dtype(instance[col]):
                instance[col] = instance[col].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

        instance = instance.replace({np.nan: None, np.inf: None, -np.inf: None})
        instance = instance.where(pd.notnull(instance), None)
        instance = instance.to_dict(orient="records")
        supabase.table('PPA').insert(instance).execute()

    def buildMark(self):
        if not self.proxy.empty:
            index = datetime_generation(self.start_date, self.end_date)

            mark_df = pd.DataFrame(index=index)
            mark_df["month"] = mark_df.index.month
            mark_df["hour"] = mark_df.index.hour

            proxy = self.proxy.copy()
            proxy["month"] = proxy.index.month
            proxy["hour"] = proxy.index.hour

            seasonal_avg = proxy.groupby(["month", "hour"])["generation"].mean().reset_index()

            mark_df = mark_df.reset_index().merge(seasonal_avg, on=["month", "hour"], how="left").set_index(
                "index")

            self.mark = mark_df[["generation"]]
            self.p50 = round(self.mark.sum().values[0], 2)
        else:
            mylogger.logger.warning('No proxy saved. Run buildProxy first.')
        return

    def computeCaptureRate(self):
        if self.proxy.empty:
            mylogger.logger.warning('No proxy saved. Run buildProxy first.')
            yearly_capture = {}
            return yearly_capture
        else:
            df_prices = getDfSupabase('DAPowerPriceFR')
            df_prices['id'] = pd.to_datetime(df_prices['id'], utc=True)
            df_prices['id'] = df_prices['id'].dt.tz_convert('Europe/Paris')

            df_prices.index = df_prices['id']

            self.proxy = self.proxy[~self.proxy.index.duplicated(keep='first')]
            df_prices = df_prices[~df_prices.index.duplicated(keep='first')]

            df = pd.concat([self.proxy, df_prices], axis=1)
            yearly_capture = {}

            for y in sorted(df.index.year.unique()):
                df_temp = df[df.index.year==y]
                avg_price = AVG_price(df_temp)
                vwavg_price = VWA_price(df_temp)
                yearly_capture[str(y)] = {'average price': avg_price, 'VWA price': vwavg_price,
                                             'Capture rate': (vwavg_price / avg_price * 100)}
            return yearly_capture

    def showPowerCurve(self):
        from Model.Power.dataProcessing import plot_hexbin_density

        weather = getDfSupabase("WeatherFR")
        weather["id"] = pd.to_datetime(weather["id"], utc=True)
        weather['id'] = weather['id'].dt.tz_convert('Europe/Paris')

        weather = (weather
                   .set_index("id")
                   .sort_index()
                   .loc[:, ["Solar_Radiation"]])
        weather = weather[~weather.index.duplicated(keep='first')]
        df = pd.concat([self.proxy, weather], axis=1)
        df = df.dropna()
        plot_hexbin_density(df, 'Solar_Radiation', 'generation')
        plt.show()

if __name__ == '__main__':

    ppa = PPA()
    ppa.buildProxy()
    ppa.showPowerCurve()
    ppa.buildMark()
    # ppa.computeCaptureRate()

    #
    # ppa.buildProxy()
    # ppa.buildMark()
    # ppa.saveInstance()

    # PPA.buildProxy()
    # PPA.saveInstance()
    # print('end')