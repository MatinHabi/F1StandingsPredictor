import pandas as pd
import plot as p
from pathlib import Path
import os
import glob
import matplotlib.pyplot as plt


def make_super_plot():
        
    file = glob.glob('./all/*.csv')

    combined = pd.concat([pd.read_csv(f) for f in file ],ignore_index=True)
    combined.to_csv('All_Races_Ever.csv', index=False)

    p.corr_with_laptime('All_Races_Ever.csv')
    p.tyre_deg('All_Races_Ever.csv')


def plot_all_in_dir(dirpath, data_dir = 'all'):
    for filename in glob.glob(f'{dirpath}/*.csv'):
        p.corr_with_laptime(filename)

def save_plots_cwl(save_dir='plots', data_dir='all'):

    os.makedirs(save_dir, exist_ok=True)

    for file_path in glob.glob(f'{data_dir}/*.csv'):
        filename = os.path.splitext(os.path.basename(file_path))[0] #gets just the file name
        p.corr_with_laptime(filename=filename,show=False, path=data_dir)

        save_path = os.path.join(save_dir, f'{filename}.png')

        plt.savefig(save_path, dpi = 300, bbox_inches='tight')
        plt.close()


def save_plots_td(save_dir='plots', data_dir='all'):

    os.makedirs(save_dir, exist_ok=True)

    for file_path in glob.glob(f'{data_dir}/*.csv'):
        filename = os.path.splitext(os.path.basename(file_path))[0] #gets just the file name
        p.tyre_deg(filename=filename,show=False, path=data_dir)

        save_path = os.path.join(save_dir, f'{filename}.png')

        plt.savefig(save_path, dpi = 300, bbox_inches='tight')
        plt.close()
