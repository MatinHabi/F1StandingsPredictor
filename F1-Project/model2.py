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

    def forward(self,driver, car, track):
        x = torch.cat([x,self.embeddings(driver, car, track)], dim = 1)
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

files = glob.glob("all2/*.csv")
d = pd.concat([pd.read_csv(f) for f in files], ignore_index = True)
d = d[['Driver','LapNumber','TyreLife','FuelLoad','Car','Track','Year','Race','TotalLaps','LapTime']]
d = d.sort_values(['Year','Race','Driver','LapNumber']).reset_index(drop = True)
d['LapNumber'] = d['LapNumber'].astype(int)

#split by year
train = d[d['Year'] <= 2023].reset_index(drop=True)
val = d[d['Year'] == 2024].reset_index(drop=True)
test = d[d['Year'] == 2025].reset_index(drop=True)

train_X , train_Y = train.drop(columns='LapTime') , train['LapTime']
val_X , val_Y = val.drop(columns='LapTime'), val['LapTime']
test_X , test_Y = test.drop(columns='LapTime'), test['LapTime']

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

PIT_LAPS = [20,40]
PIT_LOSS = 20

#getting the most rececnt pit stop
def get_last_pit(lap):
    pit = 0
    if lap < 20:
        pit = 0
    elif lap >=20 and lap < 40:
        pit = 20
    elif lap >= 40:
        pit = 40
    
    return pit


def simulate_race(model, race):
    total_laps = int(race['TotalLaps'].iloc[0])
    rows = []

    for driver, laps in race.groupby('Driver'):
        for lap in range(1 + total_laps + 1):
            rows.append({
                'Driver': driver,
                'LapNumber': lap,
                'TyreLife': 1 - ((lap - get_last_pit(lap))/ total_laps),
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
    race_time = matrix.sum(axis = 1) + len(PIT_LAPS) * PIT_LOSS #drivers true lap time + pit loss
    table = pd.DataFrame({'Driver': matrix.index, 'TotalTime': race_time.values})
    table = table.sort_values('RaceTime').reset_index(drop=True)
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


