import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import scienceplots
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from uncertainties import ufloat
from uncertainties import unumpy

plt.style.use('science')

files = glob.glob("UklSve/data/*.dat")
files.sort(key=str.lower)
print(files)

data10 = pd.read_csv(files[0], sep='\t', names=['x', 'I'])
data1 = pd.read_csv(files[2], sep='\t', names=['x', 'I'])
data2 = pd.read_csv(files[4], sep='\t', names=['x', 'I'])
data3 = pd.read_csv(files[6], sep='\t', names=['x', 'I'])
data5 = pd.read_csv(files[8], sep='\t', names=['x', 'I'])

data1['x'] = data1['x'] - data1['x'][data1['I'].idxmax()]
data10['x'] = data10['x'] - data10['x'][data10['I'].idxmax()]
data2['x'] = data2['x'] - data2['x'][data2['I'].idxmax()]
data3['x'] = data3['x'] - data3['x'][data3['I'].idxmax()]
data5['x'] = data5['x'] - data5['x'][data5['I'].idxmax()]

N = 2

def fit(x, i, a, b, c):
    return i*(np.sin(np.pi * a * x)/(np.pi * a * x))**2*(np.sin(N*b*x)/np.sin(b*x))**2 + c


popt, pcov = curve_fit(fit, data2['x'], data2['I'], [0.02, 0.01, 0.4, 0], method='trf')#deep seek cooked here

perr = np.sqrt(np.diag(pcov)) 

print("Fitted parameters with errors:")
for i, (param, err) in enumerate(zip(popt, perr)):
    print(f"Parameter {i}: {param:.3f} ± {err:.3f}")


plt.plot(data2['x'], data2['I'], 'bo', label='data')
plt.plot(data2['x'], fit(data2['x'], *[0.025, 0.032, 0.4, 0]), 'r-', label='initial guess')
plt.legend()
plt.show()