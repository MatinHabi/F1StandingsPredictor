import glob
import os
import pandas as pd
import model as m # importing loads all2/, splits the years and fits the encoder/indexes/normalising

# 1. make the model
net = m.make_model()

# 2. train on 2018-2023
m.train_model(net)

# 3. eval on 2024 - lap by lap with its own predictions + predicted standings for each race
m.run_races(net, m.val, 'eval')

# 4. test on 2025 - same as eval
m.run_races(net, m.test, 'test')
