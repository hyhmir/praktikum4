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


mojT = glob.glob('Boltz/data/mojT/*.txt')
mojU = glob.glob('Boltz/data/mojU/*.txt')
tvojU = glob.glob('Boltz/data/tvojU/*.txt')


mojT_U = [0.58, 0.5]
mojT_Uerr = [5e-4, 5e-4]

mojU_T = np.array([55.9, 36.3, 14.7]) + 273.15
mojU_Terr = [0.1, 0.3, 0.6]

tvojU_T = np.array([34, 15.3, 35.8, 22.5, 35, 56.4, 54.1, 55, 15.9, 36.4, 35, 34, 15, 17, 52.8, 54.3, 28.2, 17, 13.8, 14, 13.7, 36.6, 53, 57, 55, 55, 48, 35.8, 36, 55, 35, 54.9, 60, 56, 17]) + 273.15
tvojU_Terr = np.ones(len(tvojU_T))


def linfit(x, a, b):
    return a*x + b


ekt = []
ekterr = []


for file in mojU:
    df = pd.read_csv(file, sep='\t', names=['U', 'I'])
    df = df.query('U >=0.4')
    df['lnI'] = np.log(df['I'])


    popt, pcov = curve_fit(linfit, df['U'], df['lnI'])

    perr = np.sqrt(np.diag(pcov))


    eskbt = ufloat(popt[0], perr[0])

    ekt = ekt + [popt[0]]
    ekterr = ekterr + [perr[0]]

    u = np.linspace(0.4, df['U'].max(), 1000)

    plt.plot(df['U'], df['lnI'], 'o', ms=2, label=str(round(mojU_T[mojU.index(file)] - 273.15, 1)) + 'meritev')
    plt.plot(u, linfit(u, *popt), label=str(round(mojU_T[mojU.index(file)] - 273.15, 1)) + 'fit')

plt.legend()
plt.grid()
plt.xlabel('U [$V$]')
plt.ylabel('$ln(I_c/I_1)$')
plt.title('Graf logaritma toka proti napetosti')
plt.savefig('Boltz/porocilo/moju_lin.pdf', dpi=1024)
plt.clf()


esk = (np.array(ekt) * mojU_T)
eskerr = (np.array(ekt) * mojU_Terr)

esk_avg = np.average(esk)
esk_err = np.sqrt(np.max(eskerr)**2 + np.std(esk)**2)

print(esk, eskerr)
print(esk_avg, esk_err)



ekt = []
ekterr = []


for file in mojU + tvojU:
    df = pd.read_csv(file, sep='\t', names=['U', 'I'])
    df = df.query('U >=0.4')
    df['lnI'] = np.log(df['I'])


    popt, pcov = curve_fit(linfit, df['U'], df['lnI'])

    perr = np.sqrt(np.diag(pcov))


    eskbt = ufloat(popt[0], perr[0])

    ekt = ekt + [popt[0]]
    ekterr = ekterr + [perr[0]]


esk = (np.array(ekt) * (list(mojU_T) + list(tvojU_T)))
eskerr = (np.array(ekt) * (mojU_Terr + list(tvojU_Terr)))

esk_avg = np.average(esk)
esk_err = np.sqrt(np.max(eskerr)**2 + np.std(esk)**2)


print(esk_avg, esk_err)


