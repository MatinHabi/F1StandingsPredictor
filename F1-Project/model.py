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

## TO DO:
# ITS NOT AUTOREGRESSIVE YET. it needs the previous lap as an input feature (first lap prev laptime = 0.0) - DONE
    # add lag laps - DONE
# embeddings is lowkey broken - figure out how input data flows and make it so each driver has their own embedding + A DEFAULT EMBEDDING FOR NEW DRIVERS - DONE
        # each learnt feature gets its own n+1 x j+1 matrix (where n is number of drivers and j is number of embeddings)
# mayber random numbers inititally for embeddings instead of 0.5 ? - DONE
# make loss function - huber loss - DONE
# fix the data Huber Loss takes care of outliers so might aswell include non-green flag laps - DONE (green filter is commented out in buildDataset2)
# add the standings matrix - DONE
# optimiser = Adam - DONE
# training loop - DONE
# add pitstops for specific laps - DONE
# eval loop - DONE
# testing loop - DONE

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
    def __init__(self, dn, cn, tn, driver_dim, car_dim, track_dim, input_dim, hidden1_dim, hidden2_dim, output_dim):
        super().__init__()
        self.embeddings = Embedding(dn,cn,tn,driver_dim, car_dim, track_dim)
        self.hidden1 = nn.Linear(input_dim + driver_dim + car_dim + track_dim, hidden1_dim)
        self.hidden2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.output = nn.Linear(hidden2_dim, output_dim)

    def forward(self, x, driver, car, track):
        #lap features next to the driver/car/track embeddings
        x = torch.cat([x, self.embeddings(driver, car, track)], dim = 1)
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

#loading every race file into one dataframe
files = glob.glob("all2/*.csv")
frames = []
for f in files:
    frames.append(pd.read_csv(f))
d = pd.concat(frames, ignore_index = True)
d['LapNumber'] = d['LapNumber'].astype(int)
d = d.sort_values(['Year','Race','Driver','LapNumber']).reset_index(drop = True)

#lag lap: driver's previous lap time in the same race (first lap is NaN, becomes 0.0 after normalising)
d['PrevLapTime'] = d.groupby(['Year','Race','Driver'])['LapTime'].shift(1)

#split by year
train = d[d['Year'] <= 2023].reset_index(drop=True)
val = d[d['Year'] == 2024].reset_index(drop=True)
test = d[d['Year'] == 2025].reset_index(drop=True)

#onehot encoder for Compound and Race (Driver, Car, Track use embeddings)
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(train[['Compound','Race']])

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

#normalising columns - stats from train only
lap_mu, lap_std = train['LapNumber'].mean(), train['LapNumber'].std()
tyreLife_mu, tyreLife_std = train['TyreLife'].mean() , train['TyreLife'].std()
trackTemp_mu, trackTemp_std = train['TrackTemp'].mean(), train['TrackTemp'].std()
lapTime_mu, lapTime_std = train['LapTime'].mean(), train['LapTime'].std()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

#turning the dataframes to tensors
def to_tensor(frame):
    num = pd.DataFrame()
    num['LapNumber'] = (frame['LapNumber'] - lap_mu)/lap_std
    num['TyreLife'] = (frame['TyreLife'] - tyreLife_mu) / tyreLife_std
    num['TrackTemp'] = (frame['TrackTemp'] - trackTemp_mu) / trackTemp_std
    num['FuelLoad'] = frame['FuelLoad']
    num['PrevLapTime'] = ((frame['PrevLapTime'] - lapTime_mu) / lapTime_std).fillna(0.0)
    oneHotFeatures = encoder.transform(frame[['Compound','Race']])
    X = np.concatenate([num.values, oneHotFeatures], axis=1)
    X = torch.tensor(X,dtype=torch.float32).to(device)
    driver = torch.tensor(frame['Driver'].map(driver_index).fillna(0).astype(int).values).to(device)
    car = torch.tensor(frame['Car'].map(car_index).fillna(0).astype(int).values).to(device)
    track = torch.tensor(frame['Track'].map(track_index).fillna(0).astype(int).values).to(device)
    return X, driver, car, track

#pitstops - same laps and tyres for every driver (eval/test only, training uses the real tyres)
PIT_LAPS = [20,40]
COMPOUNDS = ['SOFT','MEDIUM','HARD']
PIT_LOSS = 20

#how many pit stops have happened before this lap (picks the compound: 0 = SOFT, 1 = MEDIUM, 2 = HARD)
def pits_done_before(lap):
    pits_done = 0
    for pit_lap in PIT_LAPS:
        if lap > pit_lap:
            pits_done += 1
    return pits_done

#the most recent pit stop before this lap (0 = no stop yet, still on the starting tyres)
def last_pit_before(lap):
    last_pit = 0
    for pit_lap in PIT_LAPS:
        if lap > pit_lap:
            last_pit = pit_lap
    return last_pit

def simulate_race(model, race):
    #real race distance worked out from fuel load, the last laps are often missing from the data
    #fuel load = (total - lap) / total, so total = lap / (1 - fuel load)
    total_laps = race['LapNumber'] / (1 - race['FuelLoad'])
    total_laps = int(round(total_laps.median()))

    #track temp per lap, filled in for laps missing from the data
    temps = race.groupby('LapNumber')['TrackTemp'].mean()
    temps = temps.reindex(range(1, total_laps + 1))
    temps = temps.ffill().bfill()

    #one row per driver per lap for the full race distance
    rows = []
    for driver, laps in race.groupby('Driver'):
        for lap in range(1, total_laps + 1):
            rows.append({
                'Driver': driver,
                'LapNumber': lap,
                'Compound': COMPOUNDS[pits_done_before(lap)],
                'TyreLife': lap - last_pit_before(lap),
                'TrackTemp': temps[lap],
                'FuelLoad': (total_laps - lap) / total_laps,
                'Car': laps['Car'].iloc[0],
                'Track': laps['Track'].iloc[0],
                'Race': laps['Race'].iloc[0],
            })

    sim = pd.DataFrame(rows)
    sim['PredLapTime'] = np.nan

    #lap by lap, each driver's predicted lap becomes their previous lap for the next one
    prev = {}
    model.eval()
    with torch.no_grad():
        for lap in range(1, total_laps + 1):
            rows_now = (sim['LapNumber'] == lap) #mask for the lap we're currently on
            now = sim[rows_now].copy() #just the rows for the lap we're on
            now['PrevLapTime'] = now['Driver'].map(prev)

            X, driver, car, track = to_tensor(now)
            pred = model(X, driver, car, track)
            pred = pred.squeeze(1).cpu().numpy()
            sim.loc[rows_now, 'PredLapTime'] = pred

            #remember each driver's prediction for the next lap
            for drv, p in zip(now['Driver'], pred):
                prev[drv] = p
    return sim

def standings(sim):
    #standings matrix: one row per driver, one column per lap, predicted lap times inside
    matrix = sim.pivot(index='Driver', columns='LapNumber', values='PredLapTime')
    #add every lap together + the pit stops, fastest race time wins
    race_time = matrix.sum(axis=1) + len(PIT_LAPS) * PIT_LOSS
    table = pd.DataFrame({'Driver': matrix.index, 'RaceTime': race_time.values})
    table = table.sort_values('RaceTime').reset_index(drop=True)
    table['Position'] = table.index + 1
    return table

def actual_result(year, race_name):
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

def run_races(model, frame, name):
    #eval/test loop: simulate every race lap by lap, print its standings, tau and the loss
    loss_total = 0
    count = 0
    taus = []
    grid_taus = []

    for (year, race_name), race in frame.groupby(['Year', 'Race']):
        sim = simulate_race(model, race)

        #loss against the real laps that exist in the data
        both = race.merge(sim[['Driver', 'LapNumber', 'PredLapTime']], on=['Driver', 'LapNumber'])
        pred = torch.tensor(both['PredLapTime'].values)
        real = torch.tensor(both['LapTime'].values)
        loss = loss_fn(pred, real)
        loss_total += loss.item() * len(both)
        count += len(both)

        #kendall's tau: predicted order vs actual order
        table = standings(sim)
        positions = table.merge(actual_result(year, race_name), on='Driver')
        tau = kendalltau(positions['Position'], positions['ActualPosition']).statistic
        taus.append(tau)

        #baseline to beat: everyone finishes where they started
        grid_tau = kendalltau(positions['GridPosition'], positions['ActualPosition']).statistic
        grid_taus.append(grid_tau)

        print(f'\n{year} {race_name}')
        print(table[['Position', 'Driver', 'RaceTime']].to_string(index=False))
        print(f'tau: {tau:.3f} (grid baseline: {grid_tau:.3f})')

    #how many races the model ordered better than the grid did
    beat_grid = 0
    for tau, grid_tau in zip(taus, grid_taus):
        if tau > grid_tau:
            beat_grid += 1

    print(f'\n{name} loss: {loss_total/count:.3f}')
    print(f'{name} mean tau: {np.mean(taus):.3f}')
    print(f'{name} mean grid baseline tau: {np.mean(grid_taus):.3f}')
    print(f'{name} races beating the grid baseline: {beat_grid}/{len(taus)}')

#training data as tensors
X_train, driver_train, car_train, track_train = to_tensor(train)
Y_train = torch.tensor(train['LapTime'].values, dtype=torch.float32).unsqueeze(1).to(device)

#settings
DRIVER_EM = 0
CAR_EM = 0
TRACK_EM = 2
LR = 0.001
EPOCHS = 20
BATCH = 256
NEW_P = 0.05 #chance a training row uses the default embedding so it learns an average driver/car/track
LAG_NOISE = 0.5 #sigma in seconds of the noise added to PrevLapTime during training
PREV_COL = 4 #PrevLapTime's column in X (see to_tensor)
has_lag_train = torch.tensor(train['PrevLapTime'].notna().values, dtype=torch.float32).to(device)
n = X_train.shape[1]
loss_fn = nn.HuberLoss()

#initialising the model
def make_model():
    model = F1MLP(len(driver_index), len(car_index), len(track_index), DRIVER_EM, CAR_EM, TRACK_EM, n, n // 2, n // 4, 1).to(device)
    model.output.bias.data.fill_(lapTime_mu) #start at the average lap time instead of 0
    return model

#training loop on the train years
def train_model(model):
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    dataset = TensorDataset(X_train, driver_train, car_train, track_train, Y_train, has_lag_train)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

    for epoch in range(EPOCHS):
        model.train()
        total = 0
        for x, driver, car, track, y, has_lag in loader:
            #noise on the lag lap so the model doesn't fully trust it (first laps with no lag stay 0.0)
            noise = torch.randn(len(x), device=device) * LAG_NOISE / lapTime_std
            x[:, PREV_COL] = x[:, PREV_COL] + noise * has_lag

            #sometimes swap in the default embedding (index 0)
            driver = torch.where(torch.rand(len(driver), device=device) < NEW_P, 0, driver)
            car = torch.where(torch.rand(len(car), device=device) < NEW_P, 0, car)
            track = torch.where(torch.rand(len(track), device=device) < NEW_P, 0, track)

            pred = model(x, driver, car, track)
            loss = loss_fn(pred, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(y)
        print(f'epoch {epoch+1}: train loss {total/len(train):.3f}')
