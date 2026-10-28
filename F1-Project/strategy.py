import glob
import pandas as pd

# teams get renamed, so map old names to the current one or each era counts as a different team
TEAM_ALIASES = {
    'Toro Rosso': 'RB', 'AlphaTauri': 'RB', 'Racing Bulls': 'RB',
    'Force India': 'Aston Martin', 'Racing Point': 'Aston Martin',
    'Renault': 'Alpine',
    'Sauber': 'Kick Sauber', 'Alfa Romeo Racing': 'Kick Sauber', 'Alfa Romeo': 'Kick Sauber',
}
DRY = ['SOFT', 'MEDIUM', 'HARD']

files = glob.glob("all2/*.csv")
d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
d['LapNumber'] = d['LapNumber'].astype(int)
d['Car'] = d['Car'].replace(TEAM_ALIASES)
d['Compound'] = d['Compound'].fillna('UNKNOWN')
d = d.sort_values(['Year', 'Race', 'Driver', 'LapNumber']).reset_index(drop=True)

# real race distance from FuelLoad (same trick as simulate_race in model.py)
d['TotalLaps'] = (d['LapNumber'] / (1 - d['FuelLoad'])).groupby([d['Year'], d['Race']]).transform('median').round()

# new stint = compound changed or tyre life went down (catches same compound swaps too)
g = d.groupby(['Year', 'Race', 'Driver'])
new_stint = (d['Compound'] != g['Compound'].shift()) | (d['TyreLife'] < g['TyreLife'].shift())
d['Stint'] = new_stint.astype(int).groupby([d['Year'], d['Race'], d['Driver']]).cumsum()

# one row per stint: compound and the lap it ended on (= the pit lap, except the last stint)
stints = (d.groupby(['Year', 'Race', 'Track', 'Car', 'Driver', 'Stint'])
            .agg(Compound=('Compound', 'first'), EndLap=('LapNumber', 'max'), TotalLaps=('TotalLaps', 'first'))
            .reset_index())

# one row per driver per race: the compound sequence and the pit laps
def summarise(s):
    return pd.Series({
        'Strategy': '-'.join(s['Compound']),
        'Stops': len(s) - 1,
        'PitLaps': list(s['EndLap'].iloc[:-1]),
        'PitPct': [round(l / s['TotalLaps'].iloc[0], 2) for l in s['EndLap'].iloc[:-1]],
        'Finished': s['EndLap'].iloc[-1] >= 0.9 * s['TotalLaps'].iloc[0],
        'Dry': s['Compound'].isin(DRY).all(),
    })

races = stints.groupby(['Year', 'Race', 'Track', 'Car', 'Driver']).apply(summarise, include_groups=False).reset_index()
# DNFs and wet races aren't real strategies
races = races[races['Finished'] & races['Dry']]

# per team per track: most used strategy, then the median pit laps of the drivers who ran it
def most_common(r):
    top = r['Strategy'].value_counts()
    best = top.index[0]
    used = r[r['Strategy'] == best]
    pit_laps = pd.DataFrame(used['PitLaps'].tolist()).median().round().astype(int).tolist()
    pit_pct = pd.DataFrame(used['PitPct'].tolist()).median().round(2).tolist()
    return pd.Series({'Strategy': best, 'TimesUsed': top.iloc[0], 'OutOf': len(r),
                      'PitLaps': pit_laps, 'PitPct': pit_pct})

team_strategy = races.groupby(['Track', 'Car']).apply(most_common, include_groups=False).reset_index()
team_strategy.to_csv('team_strategies.csv', index=False)
print(team_strategy[team_strategy['Track'] == 'Silverstone'].to_string(index=False))
