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

nj = 1
g = 10
rho1 = 1000
rho2 = 2
E = 250
pi = np.pi


data = pd.read_csv('Milik/data/data.csv')

data['v+'] = [ufloat(x, 2) for x in data['v+']]
data['v-'] = [ufloat(x, 2) for x in data['v-']]


data['r'] = unumpy.sqrt(9*nj*(data['v+'] + data['v-'])/4*g*(rho1 - rho2))
data['ne'] = np.abs((3*pi*data['r']*nj/E)*(data['v+']-data['v-']))

