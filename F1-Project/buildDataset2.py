import fastf1 as f1
import pandas as pd
import os

f1.Cache.enable_cache("fastf1_cache")

def basic_structure(circuit, driver, session_type = 'R', year = 2024):
    event = f1.get_event(year,circuit)
    session = f1.get_session(year, circuit, session_type)
    
    session.load(telemetry=False,weather=True,messages=False)
    laps = session.laps
    return df

def build_dataset(circuit='Melbourne', driver='', session_type='R', years=range(2018, 2026), name=None):

