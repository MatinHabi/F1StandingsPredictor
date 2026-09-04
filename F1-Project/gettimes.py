import fastf1 as f

f.Cache.enable_cache("VER_cache")

for i in range(1,25):
 session = f.get_session(year=2024,gp= i,identifier='R')
 session.load(telemetry=False,weather=False, messages=False)

 ver = session.laps.pick_driver('VER')
 
 print("\n\n\n" + '='*20 + f"RACE {session.event['EventName']}" + '='*20 + "\n\n\n")
 print(ver[['LapNumber', 'LapTime']].to_string())