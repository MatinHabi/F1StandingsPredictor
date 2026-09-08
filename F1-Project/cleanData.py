import pandas as pd


def zscore(filename,column):
    file = pd.read_csv(f'{filename}')
    times = file[f'{column}'].values

    file[f'{column}_Zscore'] = (times - times.mean())/times.std()

    file.to_csv('ver_austria_zscores.csv', index = False)


def iqr(filename,column):
    file = pd.read_csv(f'{filename}')

    Q1 = file[f'{column}'].quantile(0.25)
    Q3 = file[f'{column}'].quantile(0.75)
    IQR = Q3-Q1

    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR

    cleaned = file[(file[f'{column}'] >= lower_bound) & (file[f'{column}'] <= upper_bound)]

    cleaned.to_csv('ver_austria_IQR.csv', index = False)