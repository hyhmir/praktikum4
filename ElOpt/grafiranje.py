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

def fit_poli1(x, I_0, I_1, d):
    return I_0 + I_1*np.sin((np.pi/180)*x + d)**2

popt, pcov = curve_fit(fit_poli1, poli1['kot'], poli1['I'], [2.85e-4, -2.9e-4, -4.7])

perr = np.sqrt(np.diag(pcov))

print(popt, perr)

kot = np.linspace(-90, 90, 1000)


plt.title('Odvisnost toka v odvisnosti od kota polarizatorja')
plt.grid()
plt.plot(poli1['kot'], poli1['I'], 'o', ms=2, label='meritev')
plt.plot(kot, fit_poli1(kot, *popt), label='fit')
plt.grid()
plt.xlabel('kot [$^{\\circ}$]')
plt.ylabel('intenziteta [relativne enote]')
plt.legend()
plt.savefig('ElOpt/porocilo/poli1.pdf', dpi=1024)
plt.clf()


