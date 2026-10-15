import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class MLP(nn.Module):
    def __init__(self,driver_dim, car_dim, track_dim,input_dim,hidden1_dim, hidden2_dim,output_dim):
        super().__init__()
        self.driver_dim = nn.Parameter(torch.full((input_dim),0.5))
        self.car_dim = nn.Parameter(torch.full((car_dim),0.5))
        self.track_dim = nn.Parameter(torch.full((driver_dim),0.5))
        self.hidden1 = nn.Linear(track_dim, hidden1_dim)
        self.hidden2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.output = nn.Linear(hidden2_dim, output_dim)

    def forward(self,input_data):
        x = torch.relu(self.driver_dim(input_data))
        x = torch.relu(self.car_dim(x))
        x = torch.relu(self.track_dim(x))
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

d = pd.read_csv("all2/all2_Australian Grand Prix.csv")
d = d.drop(columns='Year')
Y =d.iloc[:,-1]

#Need to one-hot encode Drivers, Compound, Car, Track, Race
df = d.select_dtypes(include=['object', 'string'])
d = d.drop(columns=df.columns)
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
oneHotFeatures = pd.DataFrame(encoder.fit_transform(df))

X = pd.concat([d.iloc[:,:-1], oneHotFeatures], axis=1)


#now to normalise columns: LapNumber, TyreLife, TrackTemp

#LapNumber
lap_mu, lap_std = d['LapNumber'].mean(), d['LapNumber'].std()
d['LapNumber'] = (d['LapNumber'] - lap_mu)/lap_std
#TyreLife
tyreLife_mean, tyreLife_std = d['TyreLife'].mean(), d['TyreLife'].std()
d['TyreLife'] = (d['TyreLife'] - tyreLife_mean)/tyreLife_std
#trackTemp
trackTemp_mean, trackTemp_std = d['TrackTemp'].mean(), d['TrackTemp'].std()
d['TrackTemp'] = (d['TrackTemp'] - trackTemp_mean)/trackTemp_std


