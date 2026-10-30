import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import glob
import fastf1 as f1
from scipy.stats import kendalltau

f1.Cache.enable_cache("fastf1_cache")
f1.set_log_level('WARNING') #only print warning messages

torch.manual_seed(42)
loss_fn = nn.HuberLoss()

class Embedding(nn.Module):
    def __init__(self,dn,cn,tn,driver_dim,car_dim,track_dim):
        super().__init__()
        #embeddings matrix (2D) of size (num of drivers) x (num of embeddings) - each row is a driver with driver_dim number of embeddings
        # +1 for the default embedding for drivers, cars and tracks the model has never seen before
        self.driver_em = nn.Parameter(torch.randn(dn + 1, driver_dim) * 0.1) # *0.1 to keep embeddings small
        self.car_em = nn.Parameter(torch.randn(cn + 1, car_dim) * 0.1)
        self.track_em = nn.Parameter(torch.randn(tn + 1, track_dim) * 0.1)

    def forward(self, driver, car, track):
        #looking up the drivers embeddings and returning it
        embeddings = torch.cat([self.driver_em[driver], self.car_em[car], self.track_em[track]], dim = 1)
        return embeddings

class F1MLP(nn.Module):
    def __init__(self, dn, cn, tn, driver_dim, car_dim, track_dim, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.embeddings = Embedding(dn,cn,tn,driver_dim, car_dim, track_dim)
        self.hidden1 = nn.Linear(input_dim + driver_dim + car_dim + track_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output = nn.Linear(hidden_dim // 2 , output_dim)

    def forward(self, input_data, driver, car, track):
        x = torch.cat([input_data,self.embeddings(driver, car, track)], dim = 1)
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

files = glob.glob("all2/*.csv")
d = pd.concat([pd.read_csv(f) for f in files], ignore_index = True)
d = d[['Driver','LapNumber','Compound','TyreLife','FuelLoad','Car','Track','Year','Race','TotalLaps','LapTime']]
d = d.sort_values(['Year','Race','Driver','LapNumber']).reset_index(drop = True)
d['LapNumber'] = d['LapNumber'].astype(int)

#split by year
train = d[d['Year'] <= 2023].reset_index(drop=True)
val = d[d['Year'] == 2024].reset_index(drop=True)
test = d[d['Year'] == 2025].reset_index(drop=True)


#onehot encoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(train[['Race']])

#making the defualt embeddings for any track/driver/car the model hasnt seen before
def make_index(col):
    names = sorted(train[col].unique())
    index = {}
    for i, name in enumerate(names):
        index[name] = i + 1
    return index

driver_index = make_index('Driver')
car_index = make_index('Car')
track_index = make_index('Track')

#normalising columns
lap_mu, lap_std = train['LapNumber'].mean(), train['LapNumber'].std()
tyreLife_mu, tyreLife_std = train['TyreLife'].mean() , train['TyreLife'].std()
lapTime_mu, lapTime_std = train['LapTime'].mean(), train['LapTime'].std()

device = "cuda" if torch.cuda.is_available() else "cpu"

#turning the dataframes to tensors
def to_tensor(frame):
    num = pd.DataFrame()
    num['LapNumber'] = (frame['LapNumber'] - lap_mu)/lap_std
    num['TyreLife'] = (frame['TyreLife'] - tyreLife_mu) / tyreLife_std
    num['FuelLoad'] = frame['FuelLoad']
    oneHotFeatures = encoder.transform(frame[['Race']])
    X = np.concatenate([num.values, oneHotFeatures], axis=1)
    X = torch.tensor(X,dtype=torch.float32).to(device)
    driver = torch.tensor(frame['Driver'].map(driver_index).fillna(0).astype(int).values).to(device)
    car = torch.tensor(frame['Car'].map(car_index).fillna(0).astype(int).values).to(device)
    track = torch.tensor(frame['Track'].map(track_index).fillna(0).astype(int).values).to(device)
    return X, driver, car, track

PIT_LAPS = [20,40] #fallback for a team that has no dry finish at the track in training
PIT_LOSS = 20

#teams get renamed, so map old names to the current one or each era counts as a different team
TEAM_ALIASES = {
    'Toro Rosso': 'RB', 'AlphaTauri': 'RB', 'Racing Bulls': 'RB',
    'Force India': 'Aston Martin', 'Racing Point': 'Aston Martin',
    'Renault': 'Alpine',
    'Sauber': 'Kick Sauber', 'Alfa Romeo Racing': 'Kick Sauber', 'Alfa Romeo': 'Kick Sauber',
}
#same for tracks the data names differently in some years
TRACK_ALIASES = {'Monte Carlo': 'Monaco', 'Miami Gardens': 'Miami'}
DRY = ['SOFT', 'MEDIUM', 'HARD']

#one row per driver per race: the compound sequence and the pit laps
def summarise(s):
    return pd.Series({
        'Strategy': '-'.join(s['Compound']),
        'Stops': len(s) - 1,
        'PitLaps': list(s['EndLap'].iloc[:-1]),
        'PitPct': [round(l / s['TotalLaps'].iloc[0], 2) for l in s['EndLap'].iloc[:-1]],
        'Finished': s['EndLap'].iloc[-1] >= 0.9 * s['TotalLaps'].iloc[0],
        'Dry': s['Compound'].isin(DRY).all(),
    })

#per team per track: most used strategy, then the median pit laps of the drivers who ran it
def most_common(r):
    top = r['Strategy'].value_counts()
    best = top.index[0]
    used = r[r['Strategy'] == best]
    pit_laps = pd.DataFrame(used['PitLaps'].tolist()).median().round().astype(int).tolist()
    pit_pct = pd.DataFrame(used['PitPct'].tolist()).median().round(2).tolist()
    return pd.Series({'Strategy': best, 'TimesUsed': top.iloc[0], 'OutOf': len(r),
                      'PitLaps': pit_laps, 'PitPct': pit_pct})

#building every teams most used strategy at every track
def get_strategies(frame):
    s = frame.copy()
    s['Car'] = s['Car'].replace(TEAM_ALIASES)
    s['Track'] = s['Track'].replace(TRACK_ALIASES)
    s['Compound'] = s['Compound'].fillna('UNKNOWN')

    #new stint = compound changed or tyre life went down (catches same compound swaps too)
    g = s.groupby(['Year', 'Race', 'Driver'])
    new_stint = (s['Compound'] != g['Compound'].shift()) | (s['TyreLife'] < g['TyreLife'].shift())
    s['Stint'] = new_stint.astype(int).groupby([s['Year'], s['Race'], s['Driver']]).cumsum()

    #one row per stint: compound and the lap it ended on (= the pit lap, except the last stint)
    stints = (s.groupby(['Year', 'Race', 'Track', 'Car', 'Driver', 'Stint'])
                .agg(Compound=('Compound', 'first'), EndLap=('LapNumber', 'max'), TotalLaps=('TotalLaps', 'first'))
                .reset_index())

    races = stints.groupby(['Year', 'Race', 'Track', 'Car', 'Driver']).apply(summarise, include_groups=False).reset_index()
    #DNFs and wet races aren't real strategies
    races = races[races['Finished'] & races['Dry']]
    return races.groupby(['Track', 'Car']).apply(most_common, include_groups=False).reset_index()

team_strategy = get_strategies(train) #only the training years so the 2024/2025 strategies dont leak into eval/test

#getting the teams usual pit laps at this track, scaled to this races length
def get_pit_laps(car, track, total_laps):
    car = TEAM_ALIASES.get(car, car)
    track = TRACK_ALIASES.get(track, track)
    strategy = team_strategy[(team_strategy['Track'] == track) & (team_strategy['Car'] == car)]
    if len(strategy) == 0:
        return PIT_LAPS
    return [round(p * total_laps) for p in strategy['PitPct'].iloc[0]]

#getting the most rececnt pit stop - the pit lap is the last lap on the old tyres
def get_last_pit(lap, pits):
    pit = 0
    for p in pits:
        if lap > p:
            pit = p
    return pit


def simulate_race(model, race):
    total_laps = int(race['TotalLaps'].iloc[0])
    rows = []

    for driver, laps in race.groupby('Driver'):
        pits = get_pit_laps(laps['Car'].iloc[0], laps['Track'].iloc[0], total_laps) #each driver runs their teams usual strategy
        for lap in range(1 + total_laps + 1):
            rows.append({
                'Driver': driver,
                'LapNumber': lap,
                'TyreLife': lap - get_last_pit(lap, pits),
                'Stops': len(pits),
                'FuelLoad': (total_laps - lap) / total_laps,
                'Car': laps['Car'].iloc[0],
                'Track': laps['Track'].iloc[0],
                'Race': laps['Race'].iloc[0],
        })

    sim = pd.DataFrame(rows)

    #simulating non-autoregressively (no previousl lap time) - this gets the model to predict lap-time from driver skill & embeddings
    model.eval()
    with torch.no_grad():
        X, driver, car, track = to_tensor(sim)
        pred = model(X,driver,car,track)
        sim['PredLapTime'] = pred.squeeze(1).cpu().numpy()

    return sim

#getting the final standings from sim
def standings(sim):
    matrix = sim.pivot(index = 'Driver', columns = 'LapNumber' , values = 'PredLapTime') #changes the matrix to a Driver X LapTime instead of 1 row per driver per lap
    race_time = matrix.sum(axis = 1) + sim.groupby('Driver')['Stops'].first() * PIT_LOSS #drivers true lap time + pit loss for each of their stops
    table = pd.DataFrame({'Driver': matrix.index, 'TotalTime': race_time.values})
    table = table.sort_values('TotalTime').reset_index(drop=True)
    table['Position'] = table.index + 1
    return table

def actual_results(year, race_name):
    #real finishing order from the fastf1 cache
        session = f1.get_session(year, race_name, 'R')
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        results = session.results
        #only classified drivers (retired ones removed)
        position = pd.to_numeric(results['ClassifiedPosition'], errors='coerce')
        classified = results[position.notna()]
        #grid position for the baseline, pit lane starts (0) count as starting last
        grid = classified['GridPosition'].replace(0, 99).fillna(99)

        return pd.DataFrame({
            'Driver': classified['Abbreviation'],
            'ActualPosition': classified['ClassifiedPosition'].astype(int),
            'GridPosition': grid,
        })

#this will run the races for evaluation and testing and print the kendall tau results
def run_races(model, frame, name): 
    total_loss = 0
    count = 0
    taus = []
    grid_taus = []

    for (year, race_name), race in frame.groupby(['Year', 'Race']):
        sim = simulate_race(model,race) #gets predicted lap time from simultaion

        #becuase original table has gaps but model predicts every lap we merge the tables and only compare the rows that exist for both
        both = race.merge(sim[['Driver', 'LapNumber', 'PredLapTime']], on = ['Driver', 'LapNumber']) #holds one race and is merged with prediction for said race
        pred = torch.tensor(both['PredLapTime'].values)
        real = torch.tensor(both['LapTime'].values)
        loss = loss_fn(pred,real)
        total_loss += loss.item() * len(both) # Huber loss usually gives an average loss throughout the race * by the total laps in the season to get the real value not the average
        count += len(both) # keeps track of the number of laps in the entire season

        #kendall's taugh: predicted order vs actual order
        table = standings(sim) # final predicted standings using simulated results
        positions = table.merge(actual_results(year,race_name), on = 'Driver')
        tau = kendalltau(positions['Position'], positions['ActualPosition']).statistic #kendall tau returns object (result.statistic , result.pvalue) - .statistic is the actual tau result and .pvalue is the likelihood that result is luck
        taus.append(tau)

        #the baseline: starting grid position - aim is to be better than this
        grid_tau = kendalltau(positions['GridPosition'], positions['ActualPosition']).statistic
        grid_taus.append(grid_tau)

        print(f'\n{year} {race_name}')
        print(table[['Position', 'Driver', 'TotalTime']].to_string(index=False))
        print(f'tau: {tau:.3f} (grid baseline: {grid_tau:.3f})')

    print(f'\n{name} loss: {total_loss/count:.3f}')
    print(f'{name} mean tau: {np.mean(taus):.3f}')
    print(f'{name} mean grid baseline tau: {np.mean(grid_taus):.3f}')

X_train, driver_train, car_train, track_train = to_tensor(train)
Y_train = torch.tensor(train['LapTime'], dtype = torch.float32).unsqueeze(1).to(device)

DRIVER_EM = 100
CAR_EM = 100
TRACK_EM = 50
HIDDEN = 32
LR = 0.001
EPOCH = 20
BATCH = 256
NEW_P = 0.05 #chance a training row uses the default embedding so it learns an average driver/car/track
n = X_train.shape[1]

model = F1MLP(len(driver_index), len(car_index), len(track_index), DRIVER_EM, CAR_EM, TRACK_EM, n, HIDDEN, 1).to(device)
model.output.bias.data.fill_(lapTime_mu) #weight usually start very close to 0, whereas average laptimes are ~90 seconds. considering huber loss caps how hard each error pushes & LR is small the model wastes epochs trying to get predictions closer to ~90s. this line sets the bias to 90 so first prediction is already close to the average

optimiser = torch.optim.Adam(model.parameters(), lr = LR)
dataset = TensorDataset(X_train, driver_train, car_train, track_train, Y_train)
loader = DataLoader(dataset, batch_size=BATCH, shuffle= True)

def training():
    for e in range(EPOCH):
        model.train()
        running_loss = 0.0
        for x, d, c, t, y in loader:
            pred = model(x,d,c,t)
            loss = loss_fn(pred,y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * len(y)
        print(f'epoch: {e} || training_loss: {running_loss/len(train)}')

