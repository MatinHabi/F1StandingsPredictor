import fastf1 as f
import pandas as pd
import os

f.Cache.enable_cache("fastf1_cache")

def get_driver_laps(year, driver, circuit, session_type='R'):
    event = f.get_event(year,circuit)

    if (event['F1ApiSupport'] == False):
        with open('SKIPPED.txt', 'a') as textfile:
            print(f'skipping {year} {circuit} : no_api_support', file = textfile)
        return pd.DataFrame()

    session = f.get_session(year,circuit, session_type)

    try:
        session.load(telemetry=False,weather=True,messages=False)
        laps = session.laps
    except Exception as e:
        with open('SKIPPED.txt', 'a') as textfile:
            print(f'skipping {year} {circuit} : {e}', file = textfile)
        return pd.DataFrame()

    if laps.empty:
        with open('SKIPPED.txt', 'a') as textfile:
            print(f'skipping {year} {circuit} : session.laps_empty', file = textfile)
        return pd.DataFrame()
    
    weather = session.weather_data

    driver_session = laps.pick_driver(driver) if(driver != '') else laps
    driver_session = driver_session[driver_session['Time'].notna()]

    data = pd.merge_asof(driver_session.sort_values('Time'), weather.sort_values('Time'), on='Time')

    driver_laps = data[['Driver', 'Team', 'LapNumber', 'LapTime', 'Position',
            'Stint', 'TyreLife', 'Compound', 'FreshTyre',
            'TrackStatus', 'PitInTime', 'PitOutTime', 'IsAccurate',
            'TrackTemp', 'AirTemp', 'Humidity', 'Rainfall']].copy()

    driver_laps['IsInLap'] = driver_laps['PitInTime'].notna() #this is accounting for the lap the driver peels off to go into the pits
    driver_laps['IsOutLap'] = driver_laps['PitOutTime'].notna() #this is accounting for the likely much slower lap after the driver comes out of the pit lanee

    status = driver_laps['TrackStatus'].astype(str)
    # Flags for future reference: 1 clear, 2 yellow, 4 safety car, 5 red flag, 6 VSC deployed, 7 VSC ending
    driver_laps['IsSC'] = status.str.contains('4', regex=False) # safety car
    driver_laps['IsVSC'] = status.str.contains('6',regex=False) #Virtual Safetty car

    driver_laps['Year'] = year
    driver_laps['Circuit'] = circuit
    driver_laps['LapTime'] = driver_laps['LapTime'].dt.total_seconds()
    res = session.results.set_index('Abbreviation')
    driver_laps['StartingPosition'] = driver_laps['Driver'].map(res['GridPosition']) #mapping each driver with their starting position (which was decided before the race)
        
    return driver_laps

def add_features(df):
    df = df.sort_values(['Year','Driver','LapNumber']).copy()
    g = df.groupby(['Year','Driver'])

    # trends the laptime
    df['LapTimeDelta'] = g['LapTime'].diff()

    # short-term history to sees a trends not snapshots
    #lag{1} = t-1 lap (the lap just before), lag{2} = t-2 laps ago, lag{3} = t-3 laps ago
    for k in (1,2,3):
        df[f'LapTime_lag{k}'] = g['LapTime'].shift(k)

    # field-relative pace: how well is th driver doing on lap x compared to everyone else?  
    field_median = df.groupby(['Year','LapNumber'])['LapTime'].transform('median')
    df['PaceVsMedian'] = df['LapTime'] - field_median

    #cleaned PaceVsMedian AI ASSISTED
    clean = ~(df['IsInLap'] | df['IsOutLap'] | df['IsSC'] | df['IsVSC']) #only grab clean laps
    yearly = (df[clean]
            .groupby(['Driver','Circuit','Year'])['PaceVsMedian'].mean()
            .reset_index()
            .sort_values('Year'))
    yearly['HistPaceVsMedian'] = (yearly.groupby(['Driver','Circuit'])['PaceVsMedian']
                                    .transform(lambda s: s.shift(1).expanding().mean()))
    df = df.merge(yearly[['Driver','Circuit','Year','HistPaceVsMedian']],
                    on=['Driver','Circuit','Year'], how='left')
    # END OF AI ASSISTANCE
    
    # fuel load 1.0 at lights-out, 0.0 at the flag
    total_laps = df.groupby('Year')['LapNumber'].transform('max') 
    df['FuelLoad'] = (total_laps - df['LapNumber']) / total_laps #if there's 50 laps in a race: (50 - 20)/50 = 3/5 of fuel left

   
    #FIXING pit starts
    df['StartingPosition'] = df['StartingPosition'].replace(0,20)

    #Raw Spread varies a lot beteween 2020 and 2022 which is likely due to the groud-effect aero reset REGULATION changes in 2022
    df['GroudEffectEra'] = df['Year'] >= 2022

    #gonna shift the SC/VSC flags up one lap cuz the cars are all bunched up for approx. 1 lap after SC/VSC flag turns off
    #Fastf1 wouldn't tell me that tho so yeah...
    df['IsRestartLap'] = (df.groupby(['Year','Driver'])[['IsSC', 'IsVSC']].shift(1).any(axis=1).fillna(False))

    return df

def build_dataset(circuit='Melbourne', driver='', session_type='R', years=range(2018, 2026), name=None):
    frames = []
    for y in years:
        frames.append(get_driver_laps(y,driver,circuit,session_type))

    non_empty = []
    for df in frames:
        if not df.empty:
            non_empty.append(df)

    frames = non_empty

    if not frames:
        with open('SKIPPED.txt', 'a') as textfile:
            print(f'skipping {circuit} : no years availible', file = textfile)
        return pd.DataFrame()

    data_frame = pd.concat(frames, ignore_index=True)
    data_frame = data_frame.dropna(subset = ['LapNumber', 'LapTime', 'TyreLife', 'TrackTemp', 'Compound', 'LapTimeDelta', 'LapTime_lag1', 'LapTime_lag2', 'LapTime_lag3'])
    data_frame['Compound'] = data_frame['Compound'].replace(['ULTRASOFT','HYPERSOFT','SUPERSOFT'],'SOFT')
    data_frame = add_features(data_frame)

    
    if name is not None:
        filepath = name
    else:
        out_dir = f'./{driver}' if driver else './all'
        filepath = os.path.join(out_dir, f'{driver or "all"}_{circuit}.csv')

    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    data_frame.to_csv(filepath, index=False)
    return data_frame