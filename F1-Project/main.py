import fastf1 as f
import pandas as pd

f.Cache.enable_cache('fastf1_cache')

session = f.get_session(2024,'Austria','R')
session.load(telemetry=False,weather=False,messages=False)

ver = session.laps.pick_driver('VER')[['LapNumber', 'LapTime']]
lapT = ver['LapTime'].dt.total_seconds()
ver['LapTime'] = lapT
ver.to_csv('ver_austria.csv', index=False)

file = pd.read_csv('ver_austria.csv')
print(file.dtypes)
print(file.head())