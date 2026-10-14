import fastf1 as f1
import pandas as pd
import os

f1.Cache.enable_cache("fastf1_cache")

def basic_structure(circuit, session_type = 'R', year = 2024):
    session = f1.get_session(year, circuit, session_type)
    session.load(telemetry=False, weather=True, messages=False)

    laps = session.laps[['LapTime', 'Position', 'Driver', 'Team', 'LapNumber', 'Stint', 'Compound', 'FreshTyre', 'TrackStatus', 'IsAccurate']]
    df = pd.DataFrame(laps)
    df['LapTime'] = df['LapTime'].dt.total_seconds()

    df['PitIn'] = session.laps['PitInTime'].notna()
    df = df.sort_values(['Driver', 'LapNumber'])

    temp = pd.DataFrame(session.laps)
    temp = temp.sort_values(['Driver', 'LapNumber'])
    temp['NextPitOutTime'] = temp.groupby('Driver')['PitOutTime'].shift(-1)
    temp['PitDuration'] = (temp['NextPitOutTime'] - temp['PitInTime']).dt.total_seconds()

    df = df.merge(temp[['Driver', 'LapNumber', 'PitDuration']], on=['Driver', 'LapNumber'], how='left')
    df['PitDuration'] = df['PitDuration'].fillna(0.0)

    del temp

    start = session.laps['LapStartTime'].min()
    weather = session.weather_data[session.weather_data['Time'] <= start].iloc[[1]]
    w = pd.DataFrame(weather)
    w = w.drop(columns='Time')

    df['Rain'] = w['Rainfall'].iloc[0]

    del w

    df['OutLap'] = df.groupby('Driver')['PitIn'].shift(1, fill_value=False)
    df = df[df['TrackStatus'] == '1']
    df = df.drop(columns='IsAccurate')
    df['Compound'] = df['Compound'].replace(['ULTRASOFT','HYPERSOFT','SUPERSOFT'],'SOFT')

    df['Year'] = year

    return df

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
    df = basic_structure(circuit, session_type, years)

