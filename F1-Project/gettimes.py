import fastf1 as f
import pathlib as path
import pandas as pd

f.Cache.enable_cache("VER_cache")
frames = []

for i in range(1,25):
    session = f.get_session(year=2024,gp= i,identifier='R')
    session.load(telemetry=False,weather=False, messages=False)

    ver = session.laps.pick_driver('VER')
    
    #print("\n\n\n" + '='*20 + f"RACE {session.event['EventName']}" + '='*20 + "\n\n\n")
    #print(ver[['LapNumber', 'LapTime']].to_string())

    laps =  pd.DataFrame(ver[['DriverName','LapNumber', 'LapTime']])

    frames.append(laps)

df = pd.concat(frames, ignore_index=True)
df.dtypes
df.to_csv('data.csv', index=False)
