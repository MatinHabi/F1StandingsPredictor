import torch 
import torch.nn as nn
import torch.optim as optim
import pandas as pd

df = pd.read_csv('ver_austria.csv')

X_train = torch.tensor(df['LapNumber'].values, dtype = torch.float32).unsqueeze(1)
Y_train = torch.tensor(df['LapTime'].values, dtype=torch.float32).unsqueeze(1)

print(f"X_train.shape : {X_train.shape}")
print(f"Y_train shape : {Y_train.shape}")

x_mean, x_std = X_train.mean(), X_train.std()
y_mean, y_std = Y_train.mean(), Y_train.std()

xn = (X_train - x_mean)/x_std
yn = (Y_train - y_mean)/y_std

model = nn.Linear(1,1)
loss_fn = nn.MSELoss()
opt = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(2000):
    pred = model(xn)
    loss = loss_fn(pred, yn)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 200 == 0:
        print(f"epoch {epoch:5d}  loss {loss.item():.4f}")

w = model.weight.item()
b = model.bias.item()

slope     = w * y_std.item() / x_std.item()               
intercept = y_mean.item() + y_std.item() * b - slope * x_mean.item()
rmse      = (loss.item() ** 0.5) * y_std.item()

print(f"\n\nslope: {slope}\nintercept: {intercept}\nrawMSE: {rmse}")