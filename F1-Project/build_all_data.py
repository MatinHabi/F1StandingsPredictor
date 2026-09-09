import buildDataset as bd
import pandas as pd
import fastf1 as f1

f1.Cache.enable_cache('fastf1_cache')

events_by_name = {}

for year in range(2018, 2026):
    schedule = f1.get_event_schedule(year, include_testing=False)
    for name in schedule['EventName']:
        if name not in events_by_name:
            events_by_name[name] = []
        events_by_name[name].append(year)

for name in events_by_name:
    bd.build_dataset(name, years=events_by_name[name])