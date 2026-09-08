import torch 
import torch.nn as nn
import torch.optim as optim
import pandas as pd

file = pd.read_csv('ver_austria.csv')

x = torch.tensor(file['LapNumber'].values, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(file['LapTime'].values, dtype=torch.float32).unsqueeze(1)

#normalize values
x_mean, x_std = x.mean(), x.std()
y_mean, y_std = y.mean(), y.std()

xn = (x - x_mean)/x_std #Z-scores 
yn = (y - y_mean)/y_std

model = nn.Linear(1,1)
loss_fn = nn.MSELoss()
opt = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(2000):
    pred = model(xn) #spits out a yn
    loss = loss_fn(pred , yn)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch%200 == 0:
        print(f"epoch:{epoch:5d}, loss:{loss.item():.4f}")

w = model.weight.item()
b = model.bias.item()

slope = w * y_std.item()/x_std.item()
intercept = y_mean.item() + y_std.item()*b - slope * x_mean.item()
rmse = (loss.item() ** 0.5) * y_std.item()

print(f"\n\nslope: {slope}\nintercept: {intercept}\nrawMSE: {rmse}")