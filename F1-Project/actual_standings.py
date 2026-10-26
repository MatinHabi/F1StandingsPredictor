import pandas as pd
import fastf1 as f1
import model as m # only used for the list of eval/test races, nothing gets trained

def actual_standings(year, race_name):
    # real classified results from the fastf1 cache, same columns as m.standings()
    session = f1.get_session(year, race_name, 'R')
    session.load(laps=False, telemetry=False, weather=False, messages=False)
    results = session.results
    classified = results[pd.to_numeric(results['ClassifiedPosition'], errors='coerce').notna()]
    # winner's Time is the full race time, everyone else's is the gap to the winner
    winner_time = classified['Time'].iloc[0].total_seconds()
    gaps = classified['Time'].dt.total_seconds()
    race_time = (winner_time + gaps).astype(object)
    race_time.iloc[0] = winner_time
    # lapped drivers' Time isn't a real gap, show how many laps down they were instead (e.g. +1 Lap)
    laps_down = (classified['Laps'].iloc[0] - classified['Laps']).astype(int)
    lapped = laps_down > 0
    race_time[lapped] = [f'+{l} Lap' if l == 1 else f'+{l} Laps' for l in laps_down[lapped]]
    table = pd.DataFrame({'Position': classified['ClassifiedPosition'].astype(int), 'Driver': classified['Abbreviation'], 'RaceTime': race_time})
    return table.sort_values('Position').reset_index(drop=True)

for frame in [m.val, m.test]:
    for (year, race_name), race in frame.groupby(['Year', 'Race']):
        table = actual_standings(year, race_name)
        print(f'\n{year} {race_name}')
        print(table.to_string(index=False))
