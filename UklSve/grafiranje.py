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

# reading data

data10 = pd.read_csv(files[0], sep='\t', names=['x', 'I'])
data1 = pd.read_csv(files[2], sep='\t', names=['x', 'I'])
data2 = pd.read_csv(files[4], sep='\t', names=['x', 'I'])
data3 = pd.read_csv(files[6], sep='\t', names=['x', 'I'])
data5 = pd.read_csv(files[8], sep='\t', names=['x', 'I'])

# centering data

data1['x'] = data1['x'] - data1['x'][data1['I'].idxmax()]
data10['x'] = data10['x'] - data10['x'][data10['I'].idxmax()]
data2['x'] = data2['x'] - data2['x'][data2['I'].idxmax()]
data3['x'] = data3['x'] - data3['x'][data3['I'].idxmax()]
data5['x'] = data5['x'] - data5['x'][data5['I'].idxmax()]

# filtering data

data1 = data1[(data1['x'] >= -75) & (data1['x'] <= 75)].reset_index(drop=True)
data2 = data2[(data2['x'] >= -75) & (data2['x'] <= 75)].reset_index(drop=True)
data3 = data3[(data3['x'] >= -75) & (data3['x'] <= 75)].reset_index(drop=True)
data5 = data5[(data5['x'] >= -75) & (data5['x'] <= 75)].reset_index(drop=True)
data10 = data10[(data10['x'] >= -75) & (data10['x'] <= 75)].reset_index(drop=True)

N = 1

def fit(x, i, a, b, c):
    return i*(np.sinc(a * x))**2*(np.sinc(N*b*x)/np.sinc(b*x))**2 + c


data = [None, data1, data2, data3, None, None, data5, None, None, None, None, None, data10]


popt, pcov = curve_fit(fit, data1['x'], data1['I'], [0.086, 0.026, 0.097, 0.005], method='lm')#deep seek cooked here

perr = np.sqrt(np.diag(pcov)) 
print(popt)

print("Fitted parameters with errors:")
for i, (param, err) in enumerate(zip(popt, perr)):
    print(f"Parameter {i}: {param:.3f} ± {err:.3f}")


plt.plot(data1['x'], data1['I'], 'bo', label='podatki')
plt.plot(data1['x'], fit(data1['x'], *popt), 'r-', label='fit')
plt.legend()
plt.grid()
plt.title('Interferenčna slika ene in dveh rež')
plt.show()