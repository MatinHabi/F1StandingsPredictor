import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cleanData as cd
import numpy
import os

FEATURES = ['Driver','Team','LapNumber','LapTime','Position','Stint','TyreLife',
            'Compound','FreshTyre','TrackStatus','PitInTime','PitOutTime','IsAccurate',
            'TrackTemp','AirTemp','Humidity','Rainfall','IsInLap','IsOutLap','IsSC','IsVSC','Year',
            'StartingPosition','LapTimeDelta','LapTime_lag1','LapTime_lag2','LapTime_lag3','PaceVsMedian',
            'FuelLoad','GroudEffectEra','IsRestartLap']

COMPOUND_COLOURS = {'SOFT':'red', 'MEDIUM':'yellow', 'INTERMEDIATE': 'blue', 'WET': 'blue','HARD' : 'grey'}

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

#<------------------------------ CORROLATION WITH LAP TIME ------------------------------------------>

def corr_with_laptime(filename = 'all_Spanish Grand Prix', show=False, path = 'all'):
    file = pd.read_csv(os.path.join(path,filename) + '.csv')
    d = file[FEATURES].select_dtypes(include = ['number','bool'])
    c = d.corr()['LapTime'].drop('LapTime').sort_values() #pearson's r

    plt.figure(num = 'Feature Selection', figsize=(9,9))
    sns.barplot(x=c.values,y=c.index,hue=c.index,legend=True,palette='RdBu_r')
    plt.axvline(0,color='0.3', linewidth=0.8)#.axvline adds a grey line at x=0
    plt.xlabel('Corrolation with LapTime')
    plt.ylabel('Features')
    plt.title(f'Finding Which Features Most Impact LapTime - {filename[4:]}')
    plt.tight_layout() #.tight_layout ensures the graph is padded correctly to fit feature names into one line
    print(c)
    if show : plt.show()
    return c

#<----------------------------------------- Tyre Degredation --------------------------------------->
def tyre_deg(filename,show=False):
    file = pd.read_csv(f'{filename}')
    anomalous = file[['IsSC','IsVSC','IsInLap','IsOutLap','IsRestartLap']].any(axis=1)
    clean = file[file['IsAccurate'] & ~anomalous] #what does  & ~anomalous do?
    clean = clean[clean['Compound'].isin(['SOFT','MEDIUM','HARD'])]

    plt.figure(num='Degradation', figsize=(7, 5))
    sns.lineplot(data=clean, x='TyreLife', y='LapTime',
                 hue='Compound', hue_order=['SOFT','MEDIUM','INTERMEDIATE','HARD'],
                 palette=COMPOUND_COLOURS, errorbar=('ci', 95))
    plt.xlabel('Laps on this tyre set')
    plt.ylabel('Lap time (s)')
           
    plt.title('Tyre degradation by compound — clean racing laps only')
    plt.tight_layout()
    if show : plt.show()
    return clean 
