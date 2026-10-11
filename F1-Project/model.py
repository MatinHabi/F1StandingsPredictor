import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, input_data):
        predictions = self.network(input_data)
        return predictions

data = pd.read_csv('all_Melbourne.csv')
data = pd.get_dummies(data, columns=['Driver', 'Team', 'Compound'], dtype=float)
data = data.drop(columns=data.select_dtypes(exclude=['number', 'bool']).columns)
mask = ~(data['IsInLap'] | data['IsOutLap'] | data['IsSC'] | data['IsVSC'] | data['IsRestartLap'])
data = data[mask]

X = data.drop(columns=['LapTime', 'PaceVsMedian']).astype('float32')
Y = data['LapTime'].astype('float32')

X = (X - X.mean())/(X.std() + 1e-8)

X_t = torch.tensor(X.values, dtype = torch.float32)
Y_t = torch.tensor(Y.values, dtype = torch.float32).unsqueeze(1)

loader = DataLoader(TensorDataset(X_t, Y_t), batch_size= 64, shuffle= True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(X.shape[1], 64, 1).to(device)

loss_fn = nn.HuberLoss(delta = 1.0, reduction = 'mean')
optimizer = optim.Adam(model.parameters(), lr = 0.001)

for epoch in range(100):
    model.train()
    running_loss = 0.0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        print(f'epoch {epoch}: loss {running_loss / len(loader.dataset):.4f}')






