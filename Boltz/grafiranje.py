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
from uncertainties import umath

plt.style.use('science')


t1 = pd.read_csv('Boltz/data/14_7C_U-I.txt', sep='\t', names=['U', 'I'])
t1 = t1.query('U >=0.4')
t1['lnI'] = np.log(t1['I'])


def linfit(x, a, b):
    return a*x + b


popt, pcov = curve_fit(linfit, t1['U'], t1['lnI'])

perr = np.sqrt(np.diag(pcov))


eskbt = ufloat(popt[0], perr[0])

print(k)

print(eskbt)

