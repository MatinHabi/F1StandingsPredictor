import gettimes as gt

import datetime
import os
import fastf1 as f

def build_season(year, driver = '', session_type='R', out_dir='.'):
    schedule = f.get_event_schedule(year, include_testing=False) #DO NOT INCLUDE TESTING

    schedule[schedule['F1ApiSupport']] #F1ApiSupport - whether that specific race is done thru FastF1's live timing API or whether it came from their predecessor (pre 2018 - Ergast). Kept it to be safe

    #drop races that haven'that have not happened yet or were cancelled
    
