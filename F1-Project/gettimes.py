import fastf1 as f
import pandas as pd

f.Cache.enable_cache("fastf1_cache")
frames = []

for i in range(2018,2026):
    session = f.get_session(i, 'Austria', 'R')
    session.load(telemetry = False, weather = False, messages = False)

    ver = session.laps.pick_driver('VER')[['LapNumber', 'LapTime']]
    ver['Year'] = i
    lapT = ver['LapTime'].dt.total_seconds()
    ver['LapTime'] = lapT
    frames.append(ver)

data_frame = pd.concat(frames, ignore_index=True)

data_frame.to_csv('ver_austria.csv', index=False)


