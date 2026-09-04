import fastf1 as f
from fastf1 import plotting
import matplotlib.pyplot as plt

f.Cache.enable_cache('fastf1_cache')

session = f.get_session(2025, 'Austria', 'R')
session.load()

tbl = session.results.drop(columns=['HeadshotUrl', 'CountryCode'])
#print(f"\n\n\n{tbl.to_string()}")


cols = ['TeamName','Time','Status','Points']
#print(f"\n\n\n{session.results[cols].to_string()}")

print(f"{f.get_event_schedule(2024)[['Location']]}")
