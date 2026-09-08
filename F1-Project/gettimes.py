import fastf1 as f
import pandas as pd

f.Cache.enable_cache("fastf1_cache")

def add_stint(file):
    #file = pd.read_csv(f'{filename}')
    file = file.sort_values(['Year', 'LapNumber']).copy()
    new_stint = file.groupby('Year')['TyreLife'].diff() <= 0 #pinging T when a new stint starts
    file['Stint'] = new_stint.groupby('Year').cumsum().astype(int) + 1
    #file.to_csv('stint.csv')
    return file

def get_driver_laps(year, driver, circuit, session_type='R'):
    session = f.get_session(year,circuit, session_type)
    session.load(telemetry=False,weather=True,messages=False)
    weather = session.weather_data

    driver_session = session.laps.pick_driver(driver)
    data = pd.merge_asof(driver_session.sort_values('Time'), weather.sort_values('Time'), on='Time')

    driver_laps = data[['LapNumber', 'LapTime', 'TyreLife', 'TrackTemp', 'Compound']].copy()
    driver_laps['Year'] = year
    driver_laps['LapTime'] = driver_laps['LapTime'].dt.total_seconds()
    return driver_laps
