import fastf1 as f
import pandas as pd

f.Cache.enable_cache("fastf1_cache")

def add_stint(file):
    #file = pd.read_csv(f'{filename}')
    file = file.sort_values(['Year', 'LapNumber']).copy()
    new_stint = file.groupby('Year')['TyreLife'].diff() <= 0 #pinging T when a new stint starts
    file['Stint'] = new_stint.groupby(file['Year']).cumsum().astype(int) + 1
    #file.to_csv('stint.csv')
    return file

def get_driver_laps(year, driver, circuit, session_type='R'):
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

def build_dataset(driver='', circuit='Melbourne', session_type='R', years=range(2018, 2026), name=None):
    frames = []
    for y in years:
        frames.append(get_driver_laps(y,driver,circuit,session_type))

    data_frame = pd.concat(frames, ignore_index=True)
    data_frame = data_frame.dropna(subset = ['LapNumber', 'LapTime', 'TyreLife', 'TrackTemp', 'Compound'])
    data_frame['Compound'] = data_frame['Compound'].replace('SUPERSOFT','SOFT')
    data_frame = add_stint(data_frame)
    
    filename
    if(name != None):
        filename = name
    elif (driver != ''):
        filename = f'{driver}_{circuit}.csv'
    else:
        filename = f'all_{circuit}.csv'

    data_frame.to_csv(filname,index=False)
    return data_frame
    