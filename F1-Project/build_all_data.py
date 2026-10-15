import buildDataset2 as bd
import pandas as pd
import fastf1 as f1
import os
import time


f1.Cache.enable_cache('fastf1_cache')

events_by_name = {}

for year in range(2018, 2026):
    schedule = f1.get_event_schedule(year, include_testing=False)
    for name in schedule['EventName']:
        if name not in events_by_name:
            events_by_name[name] = []
        events_by_name[name].append(year)

for name in events_by_name:
    if os.path.exists(f'all2/all2_{name}.csv'):
        continue
    while True:
        try:
            bd.build_dataset(name, years=events_by_name[name])
            break
        except Exception as e:
            if '500 calls/h' in str(e):
                print('rate limited, sleeping 1h...')
                time.sleep(3600)
            else:
                raise