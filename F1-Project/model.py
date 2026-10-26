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
f1.set_log_level('WARNING')

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

class Embeddings(nn.Module):
    def __init__(self,dn, cn, tn, driver_dim,car_dim,track_dim):
        super().__init__()
        # +1 row so index 0 is the default embedding for new drivers/cars/tracks
        self.driver_dim = nn.Parameter(torch.randn(dn + 1, driver_dim) * 0.1 )
        self.car_dim = nn.Parameter(torch.randn(cn + 1,car_dim) * 0.1)
        self.track_dim = nn.Parameter(torch.randn(tn + 1,track_dim) * 0.1)

    def forward(self,driver, car, track):
        # look up each row's vector and stick them together
        x = torch.cat([self.driver_dim[driver], self.car_dim[car], self.track_dim[track]], dim=1)
        return x



class F1MLP(nn.Module):
    def __init__(self,dn, cn, tn, driver_dim, car_dim, track_dim,input_dim,hidden1_dim, hidden2_dim,output_dim):
        super().__init__()
        self.embeddings = Embeddings(dn, cn, tn, driver_dim,car_dim,track_dim)
        self.hidden1 = nn.Linear(input_dim + driver_dim + car_dim + track_dim, hidden1_dim)
        self.hidden2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.output = nn.Linear(hidden2_dim, output_dim)

    def forward(self,input_data, driver, car, track):
        x = torch.cat([input_data, self.embeddings(driver, car, track)], dim=1)
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

files = glob.glob("all2/*.csv")
d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
d['LapNumber'] = d['LapNumber'].astype(int)
d = d.sort_values(['Year', 'Race', 'Driver', 'LapNumber']).reset_index(drop=True)

# lag lap: driver's previous lap time in the same race (first lap is NaN, becomes 0.0 after normalising)
d['PrevLapTime'] = d.groupby(['Year', 'Race', 'Driver'])['LapTime'].shift(1)

# split by year
train = d[d['Year'] <= 2023].reset_index(drop=True)
val = d[d['Year'] == 2024].reset_index(drop=True)
test = d[d['Year'] == 2025].reset_index(drop=True)

#Need to one-hot encode Compound, Race (Driver, Car, Track use embeddings now)
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(train[['Compound', 'Race']])

# embedding index for each driver/car/track, 0 = default for anyone not seen in training
def make_index(col):
    names = sorted(train[col].unique())
    index = {}
    for i, name in enumerate(names):
        index[name] = i + 1
    return index

driver_index = make_index('Driver')
car_index = make_index('Car')
track_index = make_index('Track')

#now to normalise columns: LapNumber, TyreLife, TrackTemp, PrevLapTime - stats from train only
lap_mu, lap_std = train['LapNumber'].mean(), train['LapNumber'].std()
tyreLife_mean, tyreLife_std = train['TyreLife'].mean(), train['TyreLife'].std()
trackTemp_mean, trackTemp_std = train['TrackTemp'].mean(), train['TrackTemp'].std()
lapTime_mean, lapTime_std = train['LapTime'].mean(), train['LapTime'].std()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_tensors(frame):
    num = pd.DataFrame()
    num['LapNumber'] = (frame['LapNumber'] - lap_mu)/lap_std
    num['TyreLife'] = (frame['TyreLife'] - tyreLife_mean)/tyreLife_std
    num['TrackTemp'] = (frame['TrackTemp'] - trackTemp_mean)/trackTemp_std
    num['FuelLoad'] = frame['FuelLoad']
    num['PrevLapTime'] = ((frame['PrevLapTime'] - lapTime_mean)/lapTime_std).fillna(0.0)
    oneHotFeatures = encoder.transform(frame[['Compound', 'Race']])
    X = np.concatenate([num.values, oneHotFeatures], axis=1)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    driver = torch.tensor(frame['Driver'].map(driver_index).fillna(0).astype(int).values).to(device)
    car = torch.tensor(frame['Car'].map(car_index).fillna(0).astype(int).values).to(device)
    track = torch.tensor(frame['Track'].map(track_index).fillna(0).astype(int).values).to(device)
    return X, driver, car, track

#pitstops - same laps and tyres for every driver (eval/test only, training uses the real tyres)
PIT_LAPS = [20, 40]
COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']
PIT_LOSS = 20

def simulate_race(model, race):
    # every driver runs the full distance, model feeds itself its own previous lap
    # real race distance from FuelLoad since the last laps are often missing from the data
    total_laps = int(round((race['LapNumber'] / (1 - race['FuelLoad'])).median()))
    # track temp per lap, filled in for laps missing from the data
    temps = race.groupby('LapNumber')['TrackTemp'].mean().reindex(range(1, total_laps + 1)).ffill().bfill()
    rows = []
    for driver, laps in race.groupby('Driver'):
        for lap in range(1, total_laps + 1):
            pits_done = sum(lap > p for p in PIT_LAPS)
            last_pit = max([0] + [p for p in PIT_LAPS if lap > p])
            rows.append({
                'Driver': driver,
                'LapNumber': lap,
                'Compound': COMPOUNDS[pits_done],
                'TyreLife': lap - last_pit,
                'TrackTemp': temps[lap],
                'FuelLoad': (total_laps - lap) / total_laps,
                'Car': laps['Car'].iloc[0],
                'Track': laps['Track'].iloc[0],
                'Race': laps['Race'].iloc[0],
            })
    sim = pd.DataFrame(rows)
    sim['PredLapTime'] = np.nan

    prev = {}
    model.eval()
    with torch.no_grad():
        for lap in range(1, total_laps + 1):
            rows_now = sim['LapNumber'] == lap
            now = sim[rows_now].copy()
            now['PrevLapTime'] = now['Driver'].map(prev)
            X, driver, car, track = to_tensors(now)
            pred = model(X, driver, car, track).squeeze(1).cpu().numpy()
            sim.loc[rows_now, 'PredLapTime'] = pred
            for drv, p in zip(now['Driver'], pred):
                prev[drv] = p
    return sim

def standings(sim):
    # standings matrix: one row per driver, one column per lap, predicted lap times inside
    matrix = sim.pivot(index='Driver', columns='LapNumber', values='PredLapTime')
    # add every lap together + the pit stops, fastest race time wins
    race_time = matrix.sum(axis=1) + len(PIT_LAPS) * PIT_LOSS
    table = pd.DataFrame({'Driver': matrix.index, 'RaceTime': race_time.values})
    table = table.sort_values('RaceTime').reset_index(drop=True)
    table['Position'] = table.index + 1
    return table

def actual_result(year, race_name):
    # real finishing order from the fastf1 cache, only classified drivers (retired ones removed)
    session = f1.get_session(year, race_name, 'R')
    session.load(laps=False, telemetry=False, weather=False, messages=False)
    results = session.results
    classified = results[pd.to_numeric(results['ClassifiedPosition'], errors='coerce').notna()]
    # grid position too for the baseline, pit lane starts (0) count as starting last
    grid = classified['GridPosition'].replace(0, 99).fillna(99)
    return pd.DataFrame({'Driver': classified['Abbreviation'], 'ActualPosition': classified['ClassifiedPosition'].astype(int), 'GridPosition': grid})

def run_races(model, frame, name):
    # eval/test loop: simulate every race lap by lap, print its standings, tau and the loss
    loss_total = 0
    count = 0
    taus = []
    grid_taus = []
    for (year, race_name), race in frame.groupby(['Year', 'Race']):
        sim = simulate_race(model, race)
        # loss against the real laps that exist in the data
        both = race.merge(sim[['Driver', 'LapNumber', 'PredLapTime']], on=['Driver', 'LapNumber'])
        loss = loss_fn(torch.tensor(both['PredLapTime'].values), torch.tensor(both['LapTime'].values))
        loss_total += loss.item() * len(both)
        count += len(both)
        # kendall's tau: predicted order vs actual order for the classified drivers
        table = standings(sim)
        positions = table.merge(actual_result(year, race_name), on='Driver')
        tau = kendalltau(positions['Position'], positions['ActualPosition']).statistic
        taus.append(tau)
        # baseline to beat: predict everyone finishes where they started
        grid_tau = kendalltau(positions['GridPosition'], positions['ActualPosition']).statistic
        grid_taus.append(grid_tau)
        print(f'\n{year} {race_name}')
        print(table[['Position', 'Driver', 'RaceTime']].to_string(index=False))
        print(f'tau: {tau:.3f} (grid baseline: {grid_tau:.3f})')
    print(f'\n{name} loss: {loss_total/count:.3f}')
    print(f'{name} mean tau: {np.mean(taus):.3f}')
    print(f'{name} mean grid baseline tau: {np.mean(grid_taus):.3f}')
    print(f'{name} races beating the grid baseline: {sum(t > g for t, g in zip(taus, grid_taus))}/{len(taus)}')

X_train, driver_train, car_train, track_train = to_tensors(train)
Y_train = torch.tensor(train['LapTime'].values, dtype=torch.float32).unsqueeze(1).to(device)

#now to initialise the model
DRIVER_EM = 5
CAR_EM = 2
TRACK_EM = 2
LR = 0.001
EPOCHS = 20
BATCH = 256
NEW_P = 0.05 # chance a training row uses the default embedding so it learns an average driver/car/track
LAG_NOISE = 0.5 # sigma in seconds of the noise added to PrevLapTime during training
PREV_COL = 4 # PrevLapTime's column in X (see to_tensors)
has_lag_train = torch.tensor(train['PrevLapTime'].notna().values, dtype=torch.float32).to(device)
n = X_train.shape[1]
loss_fn = nn.HuberLoss()

def make_model():
    model = F1MLP(len(driver_index), len(car_index), len(track_index), DRIVER_EM,CAR_EM,TRACK_EM,n,n//2,n//4,1).to(device)
    # start the output at the average lap time instead of 0
    model.output.bias.data.fill_(lapTime_mean)
    return model

def train_model(model):
    # training loop on the train years
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(X_train, driver_train, car_train, track_train, Y_train, has_lag_train), batch_size=BATCH, shuffle=True)
    for epoch in range(EPOCHS):
        model.train()
        total = 0
        for x, driver, car, track, y, has_lag in loader:
            # noise on the lag lap so the model doesn't fully trust it (first laps with no lag stay 0.0)
            x[:, PREV_COL] = x[:, PREV_COL] + torch.randn(len(x), device=device) * LAG_NOISE / lapTime_std * has_lag
            # sometimes swap in the default embedding
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
