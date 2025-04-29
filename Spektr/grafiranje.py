import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import scienceplots
import scipy

plt.style.use('science')

files = glob.glob("Spektr/data/*.csv")
files.sort(key=str.lower)

calibration = files[0]
cali = pd.read_csv(calibration)
cali1 = pd.DataFrame({
    'kot': pd.concat([cali['Hg kot'], cali['H2 kot']], ignore_index=True),
    'val': pd.concat([cali['Hg val'], cali['H2 val']], ignore_index=True)
})

def fit(x, c1, c2, c3):
    return c1 + c2*x + c3*x**0.5


