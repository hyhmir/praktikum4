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


poli1 = pd.read_csv('ElOpt/data/90do-90-1poli.txt', sep='\t', names=['kot', 'I'])
poli1['kot'] = 90 - poli1['kot']*5

plt.plot(poli1['kot'], poli1['I'])
plt.show()
