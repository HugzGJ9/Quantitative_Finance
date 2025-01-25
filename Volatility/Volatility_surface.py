import os
import pandas as pd

try:
    Vol_surface = pd.read_excel('Volatility/Smile.xlsx', sheet_name='smile_NG')
except FileNotFoundError:
    os.chdir('..')
    Vol_surface = pd.read_excel('Volatility/Smile.xlsx', sheet_name='smile_NG')