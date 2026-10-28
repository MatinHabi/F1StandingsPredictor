import fastf1 as f1
import pandas as pd
import os
from fastf1.exceptions import DataNotLoadedError

f1.Cache.enable_cache("fastf1_cache")

# FastF1 names the same venue differently across seasons
LOCATION_ALIASES = {'Yas Island': 'Yas Marina', 'Marina Bay': 'Singapore'}

# first year of a layout change big enough to shift lap times
LAYOUT_CHANGES = {'Melbourne': 2022, 'Yas Marina': 2021, 'Barcelona': 2023, 'Singapore': 2023}

def track_key(session, year):
    # key by venue + layout, not event name (Styrian = Austrian, Mexican = Mexico City, ...)
    if session.event['EventName'] == 'Sakhir Grand Prix':
        return 'Sakhir Outer'
    loc = session.event['Location']
    loc = LOCATION_ALIASES.get(loc, loc)
    if loc in LAYOUT_CHANGES:
        era = 'new' if year >= LAYOUT_CHANGES[loc] else 'old'
        return f'{loc}_{era}'
    return loc


def basic_structure(circuit, session_type='R', year=2024):
    session = f1.get_session(year, circuit, session_type)
    session.load(telemetry=False, weather=True, messages=False)

    try:
        laps = pd.DataFrame(session.laps)
    except DataNotLoadedError:
        return pd.DataFrame()

    laps = laps.sort_values(['Driver', 'LapNumber'])
    total_laps = laps['LapNumber'].max()   # before filtering, so fuel load is correct

    # restart lap = previous lap was under SC/VSC (cars still bunched up)
    laps['PrevTrackStatus'] = laps.groupby('Driver')['TrackStatus'].shift(1)
    laps['Compound'] = laps['Compound'].replace(['ULTRASOFT', 'HYPERSOFT', 'SUPERSOFT'], 'SOFT')

    # track temp at the time each lap ended
    weather = session.weather_data[['Time', 'TrackTemp']].sort_values('Time')
    laps = pd.merge_asof(laps.dropna(subset=['Time']).sort_values('Time'), weather, on='Time', direction='backward')

    #df = laps.loc[green, ['Driver', 'Team', 'LapNumber', 'Compound', 'TyreLife', 'TrackTemp', 'LapTime']].copy()
    df = laps.loc[:,['Driver', 'Team', 'LapNumber', 'Compound', 'TyreLife', 'TrackTemp', 'LapTime']]
    df['LapTime'] = df['LapTime'].dt.total_seconds()

    # fuel load 1.0 at lights-out, 0.0 at the flag
    df['FuelLoad'] = (total_laps - df['LapNumber']) / total_laps
    df['TotalLaps'] = total_laps
    # embedding keys
    df['Car'] = df['Team']
    df['Track'] = track_key(session, year)
    df['Year'] = year
    df['Race'] = f"{session.event['EventName']}"

    # target last
    df = df.drop(columns='Team').sort_values(['Driver', 'LapNumber'])
    return df[[c for c in df.columns if c != 'LapTime'] + ['LapTime']]


def build_dataset(circuit='Melbourne', driver='', session_type='R', years=range(2018, 2027), name=None):
    frames = []
    for y in years:
        frames.append(basic_structure(circuit,session_type,y))

    non_empty = []
    for df in frames:
        if not df.empty:
            non_empty.append(df)

    frames = non_empty

    if not frames:
        return pd.DataFrame()

    data_frame = pd.concat(frames, ignore_index=True)
    data_frame = data_frame.dropna(subset = ['LapTime', 'TyreLife', 'Compound'])

    if name is not None:
        filepath = name
    else:
        out_dir = f'./{driver}' if driver else './all2'
        filepath = os.path.join(out_dir, f'{driver or "all2"}_{circuit}.csv')

    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    data_frame.to_csv(filepath, index=False)
    return data_frame


