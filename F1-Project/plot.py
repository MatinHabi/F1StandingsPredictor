import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cleanData as cd
import numpy

# <--------------------------- SCATTER --------------------------->
def scatter(file):
    plt.figure(num="Scatter")

    sns.scatterplot(data=file,x='LapNumber',y='LapTime',hue='Year',palette='viridis')

    plt.xlabel("LapNumber")
    plt.ylabel("LapTime")
    plt.legend(title='Year',bbox_to_anchor=(1.05,1),loc='upper right')

#< --------------- Z-score ------------>
def zscore(file):
    plt.figure(num = "Z-score")

    cleaned = file[(file['LapTime_Zscore'] < 2) & (file['LapTime_Zscore'] > -2)]

    sns.scatterplot(data=cleaned,x='LapNumber',y='LapTime',hue='Year',palette='viridis')
    plt.legend(title='Year',bbox_to_anchor=(1.05,1),loc='upper right')

#<-------------- IQR ----------------->
def iqr(file):
    plt.figure(num = "IQR")

    sns.scatterplot(data = file, x = 'LapNumber', y = 'LapTime', hue = 'Year', palette = 'viridis')
    plt.legend(title= 'Year', bbox_to_anchor=(1.05,1), loc='upper right')


