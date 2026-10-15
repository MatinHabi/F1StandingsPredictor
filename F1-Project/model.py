import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

## TO DO:
# embeddings is lowkey broken - figure out how input data flows and make it so each driver has their own embedding + A DEFAULT EMBEDDING FOR NEW DRIVERS 
        # each learnt feature gets its own n+1 x j+1 matrix (where n is number of drivers and j is number of embeddings)
# mayber random numbers inititally for embeddings instead of 0.5 ?
# make loss function - huber loss
# fix the data Huber Loss takes care of outliers so might aswell include non-green flag laps
# optimiser = Adam
# ITS NOT AUTOREGRESSIVE YET. it needs the previous lap as an input feature (first lap prev laptime = 0.0)
# add the standings matrix
# training loop
# add pitstops for specific laps
# eval loop
# testing loop

class Embeddings(nn.Module):
    def __init__(self,driver_dim,car_dim,track_dim):
        super().__init__()
        self.driver_dim = nn.Parameter(torch.full((driver_dim,),0.5))
        self.car_dim = nn.Parameter(torch.full((car_dim,),0.5))
        self.track_dim = nn.Parameter(torch.full((track_dim,),0.5))

    def forward(self,x):
        x = self.driver_dim(x)
        x = self.car_dim(x)
        x = self.track_dim(x)
        return x
    


class F1MLP(nn.Module):
    def __init__(self,driver_dim, car_dim, track_dim,input_dim,hidden1_dim, hidden2_dim,output_dim):
        super().__init__()
        self.embeddings = Embeddings(driver_dim,car_dim,track_dim)
        self.hidden1 = nn.Linear(input_dim, hidden1_dim)
        self.hidden2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.output = nn.Linear(hidden2_dim, output_dim)

    def forward(self,input_data):
        x = self.embeddings(input_data)
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

#now to initialise the model
DRIVER_EM = 2
CAR_EM = 2
TRACK_EM = 2
LR = 0.001
n = len(X)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = F1MLP(DRIVER_EM,CAR_EM,TRACK_EM,n,n/2,n/4,1).to(device)


