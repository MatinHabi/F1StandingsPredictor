import fastf1 as f
import pandas as pd
import os

f.Cache.enable_cache("fastf1_cache")

def add_stint(file):
    #file = pd.read_csv(f'{filename}')
    file = file.sort_values(['Year', 'LapNumber']).copy()
    new_stint = file.groupby('Year')['TyreLife'].diff() <= 0 #pinging T when a new stint starts
    file['Stint'] = new_stint.groupby(file['Year']).cumsum().astype(int) + 1
    #file.to_csv('stint.csv')
    return file

def get_driver_laps(year, driver, circuit, session_type='R'):
    event = f.get_event(year,circuit)
    if (event['F1ApiSupport'] == False):
        return pd.DataFrame()
    
    session = f.get_session(year,circuit, session_type)
    session.load(telemetry=False,weather=True,messages=False)
    weather = session.weather_data

    driver_session = session.laps.pick_driver(driver) if(driver != '') else  session.laps

    data = pd.merge_asof(driver_session.sort_values('Time'), weather.sort_values('Time'), on='Time')

    driver_laps = data[['Driver', 'Team', 'LapNumber', 'LapTime', 'Position',
            'Stint', 'TyreLife', 'Compound', 'FreshTyre',
            'TrackStatus', 'PitInTime', 'PitOutTime', 'IsAccurate',
            'TrackTemp', 'AirTemp', 'Humidity', 'Rainfall']].copy()
    
    driver_laps['Year'] = year
    driver_laps['LapTime'] = driver_laps['LapTime'].dt.total_seconds()
    res = session.results.set_index('Abbreviation')
    driver_laps['StartingPosition'] = driver_laps['Driver'].map(res['GridPosition']) #mapping each driver with their starting position (which was decided before the race)
        
    return driver_laps

def build_dataset(circuit='Melbourne', driver='', session_type='R', years=range(2018, 2026), name=None):
    frames = []
    for y in years:
        frames.append(get_driver_laps(y,driver,circuit,session_type))

    non_empty = []
    for df in frames:
        if not df.empty:
            non_empty.append(df)

    frames = non_empty

    data_frame = pd.concat(frames, ignore_index=True)
    data_frame = data_frame.dropna(subset = ['LapNumber', 'LapTime', 'TyreLife', 'TrackTemp', 'Compound'])
    data_frame['Compound'] = data_frame['Compound'].replace('SUPERSOFT','SOFT')
    data_frame = add_stint(data_frame)
    
    if name is not None:
        filepath = name
    else:
        out_dir = f'./{driver}' if driver else './all'
        filepath = os.path.join(out_dir, f'{driver or "all"}_{circuit}.csv')

    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    data_frame.to_csv(filepath, index=False)
    return data_frame