import pandas as pd
from Graphics.Graphics import plot_power_correl_vsCountries
from Logger.Logger import mylogger
def computeCorrel(df:pd.DataFrame, index, neighbour_country):
    correl = df.corr()
    correl = correl[neighbour_country]
    correl = correl.drop(['Date (TC+1)', 'ANNEE', 'NUM MOIS', 'NUM SEMAINE', 'Date short']+neighbour_country)
    if index:
        correl.index = [index]
    return correl

def loop_computeCorrel(data, loop, neighbour_country):
    loop_correl = pd.DataFrame(columns=neighbour_country)
    for i in data[loop].unique():
        data_week = data[data[loop] == i]
        loop_correl_temp = computeCorrel(data_week, i, neighbour_country)
        if isinstance(loop_correl_temp, pd.DataFrame):
            loop_correl = pd.concat([loop_correl, loop_correl_temp], ignore_index=True)
        else:
            loop_correl = pd.concat([loop_correl, pd.DataFrame([loop_correl_temp])], ignore_index=True)
    return loop_correl

def getLocCorrel(country):
    data = pd.read_excel('EUhourlydaprice.xlsx', sheet_name='data')
    neighbour_country = ['Day Ahead Auction (FR)',
                         'Day Ahead Auction (ES)',
                         'Day Ahead Auction (DE-LU)',
                         'Day Ahead Auction (PT)']
    neighbour_country.remove(country)
    yearly_correl = computeCorrel(data, 'yearly correl', neighbour_country)
    weekly_correl = loop_computeCorrel(data, 'NUM SEMAINE', neighbour_country)
    monthly_correl = loop_computeCorrel(data, 'NUM MOIS', neighbour_country)

    plot_power_correl_vsCountries(weekly_correl, country, 'week')
    plot_power_correl_vsCountries(monthly_correl, country, 'month')

    mylogger.logger.info(yearly_correl.to_string())
    return yearly_correl.to_string()


if __name__ == '__main__':
    # 'Day Ahead Auction (ES)'
    # 'Day Ahead Auction (FR)'
    # 'Day Ahead Auction (DE-LU)'
    # 'Day Ahead Auction (PT)'
    getLocCorrel('Day Ahead Auction (ES)')
    # getLocCorrel('Day Ahead Auction (FR)')