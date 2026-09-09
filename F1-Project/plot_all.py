import pandas as pd
import plot as p
from pathlib import Path
import glob

file = glob.glob('./all/*.csv')

combined = pd.concat([pd.read_csv(f) for f in file ],ignore_index=True)
combined.to_csv('All_Races_Ever.csv', index=False)

p.corr_with_laptime('All_Races_Ever.csv')
p.tyre_deg('All_Races_Ever.csv')