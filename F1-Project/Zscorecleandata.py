import pandas as pd

file = pd.read_csv('ver_austria.csv')
times = file['LapTime'].values

file['LapTime_Zscore'] = (times - times.mean())/times.std()

file.to_csv('ver_austria_zscors.csv', index = False)


