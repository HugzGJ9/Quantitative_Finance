import pandas as pd
def keep_after_dash(s):
    parts = s.split('-')
    return parts[1] if len(parts) > 1 else s

def keep_before_dash(s):
    parts = s.split('-')
    return pd.to_datetime(parts[0], format='%y%m%d') if len(parts) > 1 else pd.to_datetime(s, format='%y%m%d')