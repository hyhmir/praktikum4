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
    if N ==1:
        return i*(np.sinc(a * x))**2 + c
    return i*(np.sinc(a * x))**2*(np.sinc(N*b*x)/np.sinc(b*x))**2 + c


data = [['mjav', 'mjav'], data1, data2, data3, ['mjav', 'mjav'], data5, ['mjav', 'mjav'], ['mjav', 'mjav'], ['mjav', 'mjav'], ['mjav', 'mjav'], data10] # debilno i know, ampak the only way this shit works


pari = [(1,2), (2,3), (3,5), (5,10)]

for par in pari:


    plt.title(f'Interferenčna slika {par[0]} in {par[1]} rež')

    N = par[0]
    df = pd.DataFrame(data[N])
    popt, pcov = curve_fit(fit, df['x'], df['I'], [0.086, 0.026, 0.097, 0.005], method='lm')

    perr = np.sqrt(np.diag(pcov)) 

    print(f"Fitted parameters with errors: {N}")
    for i, (param, err) in enumerate(zip(popt, perr)):
        print(f"Parameter {i}: {param:.3f} ± {err:.3f}")

    plt.plot(df['x'], df['I'], 'o', ms=1, label=f'podatki {N}')
    plt.plot(df['x'], fit(df['x'], *popt), 'r-', label=f'fit {N}')

    N = par[1]
    df = pd.DataFrame(data[N])
    popt, pcov = curve_fit(fit, df['x'], df['I'], [0.086, 0.026, 0.097, 0.005], method='lm')

    perr = np.sqrt(np.diag(pcov)) 

    print(f"Fitted parameters with errors: {N}")
    for i, (param, err) in enumerate(zip(popt, perr)):
        print(f"Parameter {i}: {param:.3f} ± {err:.3f}")

    plt.plot(df['x'], df['I'], 'o', ms=1, label=f'podatki {N}')
    plt.plot(df['x'], fit(df['x'], *popt), 'grey', label=f'fit {N}')

    plt.legend()
    plt.grid()
    plt.xlabel('premik v $x$ smeri od središča [mm]')
    plt.ylabel('tok $I$ na fotodiodi [A]')
    plt.savefig(f'UklSve/porocilo/uklon{par[0]}{par[1]}.pdf', dpi=1024)
    plt.clf()



