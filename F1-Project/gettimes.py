import fastf1 as f
import pandas as pd

f.Cache.enable_cache("fastf1_cache")
frames = []

for i in range(2018,2026):
    session = f.get_session(i, 'Austria', 'R')
    session.load(telemetry = False, weather = True, messages = False)

    weather = session.weather_data #we need this cuz track_temp isn't part of session
    driver_session = session.laps.pick_driver('VER')
    laps = pd.merge_asof(driver_session.sort_values('Time'), weather.sort_values('Time'), on='Time') #this merges the track temp data to the closest time stamp

    driver_laps = laps[['LapNumber','LapTime','TyreLife','TrackTemp','Compound']].copy()
    driver_laps['Year'] = i
    lapT = driver_laps['LapTime'].dt.total_seconds()
    driver_laps['LapTime'] = lapT
    frames.append(driver_laps)

data_frame = pd.concat(frames, ignore_index=True)
data_frame = data_frame.dropna(subset = ['LapNumber','LapTime','TyreLife','TrackTemp','Compound']) #dropping NaNs
data_frame['Compound'] = data_frame['Compound'].replace('SUPERSOFT', 'SOFT') # replacing SUPERSOFT with SOFT for 2018
data_frame.to_csv('ver_austria_noNaN.csv', index=False)


