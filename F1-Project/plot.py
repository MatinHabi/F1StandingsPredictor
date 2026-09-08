import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy

file = pd.read_csv('ver_austria.csv')

# <--------------------------- SCATTER --------------------------->
plt.figure(num="Scatter")
x_data = file['LapTime'].values
y_data = file['LapNumber'].values
year = file['Year'].values

sns.scatterplot(data=file,x=x_data,y=y_data,hue=year,palette='viridis')

plt.xlabel("LapNumber")
plt.ylabel("LapTime")
plt.legend(title='Year',bbox_to_anchor=(1.05,1),loc='upper right')

# <--------------------------- BAR --------------------------->
"""
plt.figure(num="Bar")
plt.bar(x_data,y_data,color='red', edgecolor='black')
plt.xlabel("LapNumber")
plt.ylabel("LapTime")

"""
# <----- Normalized data ------ >
plt.figure(num="Normalized")
x = x_data
y = y_data

x_mean, x_std = x.mean(), x.std()
y_mean, y_std = y.mean(), y.std()

xn = (x - x_mean)/x_std
yn = (y - y_mean)/y_std

sns.scatterplot(data=file,x=xn,y=yn,hue=year,palette='viridis')

plt.title("NORMALISED")
plt.xlabel("LapTime")
plt.ylabel("LapNumber")
plt.legend(title='Year',bbox_to_anchor=(1.05,1),loc='upper right')

plt.show()

#< --------------- CLEANED 
