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


COMPOUND_ORDER   = ['SOFT', 'MEDIUM', 'INTERMEDIATE' , 'HARD']
COMPOUND_COLOURS = {'SOFT': '#C81E1E','MEDIUM': '#E1A100','HARD': '#3B7DD8', 'INTERMEDIATE': '#2CA02C', 'WET': '#1F4E8C',}
NUMERIC          = ['LapNumber', 'LapTime', 'TyreLife', 'TrackTemp']

def race_trace(filename):
    plt.figure(num="new shit")
    file = pd.read_csv(f'{filename}')

    g = sns.relplot(data=file, x='LapNumber', y='LapTime',
                    hue='Compound', hue_order=COMPOUND_ORDER, palette=COMPOUND_COLOURS,
                    style='Compound', style_order=COMPOUND_ORDER,
                    col='Year', col_wrap=4, height=2.5, aspect=1.15,
                    s=28, alpha=0.9, edgecolor='white', linewidth=0.4)

    for year, ax in g.axes_dict.items():
        year_data = file[file['Year'] == year]
        for _, stint in year_data.groupby('Stint'):
            ax.plot(stint['LapNumber'], stint['LapTime'],
                    color='0.55', linewidth=1, alpha=0.6, zorder=0)
        for stop in year_data.groupby('Stint')['LapNumber'].min()[1:]:
            ax.axvline(stop - 0.5, color='0.8', linestyle='--', linewidth=1, zorder=0)

    g.set_titles("{col_name}")
    g.set_axis_labels("Lap", "Lap time (s)")
    g.figure.suptitle("Race trace by season - dashed lines are pit stops", y=1.02)
    sns.move_legend(g, 'center right', title='Compound', frameon=False)

    plt.show()
    return g