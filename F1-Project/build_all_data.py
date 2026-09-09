import buildDataset as bd
import pandas as pd
import fastf1 as f1

f1.Cache.enable_cache('fastf1_cache')

years_by_location = {}

for year in range(2018, 2026):
    schedule = f1.get_event_schedule(year, include_testing=False)
    locations = schedule['Location']

    for loc in locations:
        if loc not in years_by_location:
            years_by_location[loc] = []
        years_by_location[loc].append(year)

for loc in years_by_location:
    years = years_by_location[loc]
    bd.build_dataset(loc, years=years)