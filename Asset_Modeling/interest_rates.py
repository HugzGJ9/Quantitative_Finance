import yfinance as yf

Tresury_bond_13weeks = yf.Ticker('^IRX').history().tail(1)['Close'].values[0]/ 100
Tresury_bond_5years = yf.Ticker('^FVX').history().tail(1)['Close'].values[0]/ 100
Tresury_bond_30years = yf.Ticker('^TYX').history().tail(1)['Close'].values[0]/ 100
