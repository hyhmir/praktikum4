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


### prvi poli

poli1 = pd.read_csv('ElOpt/data/90do-90-1poli.txt', sep='\t', names=['kot', 'I'])
poli1['kot'] = 90 - poli1['kot']*5

def fit_poli1(x, I_0, I_1, d):
    return I_0 + I_1*np.sin((np.pi/180)*x + d)**2

popt, pcov = curve_fit(fit_poli1, poli1['kot'], poli1['I'], [2.85e-4, -2.9e-4, -4.7])

perr = np.sqrt(np.diag(pcov))

print('prvi poli:')
print(popt, perr)

kot = np.linspace(-90, 90, 1000)


plt.plot(poli1['kot'], poli1['I'], 'o', ms=2, label='meritev')
plt.plot(kot, fit_poli1(kot, *popt), label='fit')
plt.title('Odvisnost toka v odvisnosti od kota polarizatorja')
plt.xlabel('kot [$^{\\circ}$]')
plt.ylabel('intenziteta [relativne enote]')
plt.grid()
plt.legend(loc='upper left')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.savefig('ElOpt/porocilo/poli1.pdf', dpi=1024)
plt.clf()


### drugi poli

poli2_1 = pd.read_csv('ElOpt/data/-90do90-2poli.txt', sep='\t', names=['kot', 'damn'])
poli2_2 = pd.read_csv('ElOpt/data/90do-90-2poli.txt', sep='\t', names=['kot', 'damn'])
poli2 = pd.DataFrame()
poli2['kot'] = poli2_1['kot']
poli2['damn'] = poli2_1['damn'] + poli2_2['damn'] # zgolj nekaj lepšanja podatkov, dont sue me
poli2['kot'] = 90 - poli2['kot']*5

def fit_poli2(x, I_0, I_1, d):
    return I_0 + I_1*np.sin(2*(np.pi/180)*x + d)**2

popt, pcov = curve_fit(fit_poli2, poli2['kot'], poli2['damn'])

perr = np.sqrt(np.diag(pcov))

print('drugi poli:')
print(popt, perr)

plt.plot(poli2['kot'], poli2['damn'], 'o', ms=2, label='meritve')
plt.plot(kot, fit_poli2(kot, *popt), label='fit')
plt.title('Odvisnost toka v odvisnosti od kota polarizatorja')
plt.grid()
plt.xlabel('kot [$^{\\circ}$]')
plt.ylabel('itenziteta [relativne enote]')
plt.legend(loc='upper left')
plt.savefig('ElOpt/porocilo/poli2.pdf', dpi=1024)
plt.clf()


### kerr

kerr = pd.read_csv('ElOpt/data/moč-v-napetost1-keramika.txt', sep='\t', names=['V', 'P'])
kerr['V'] = pd.read_csv('ElOpt/data/napetost1.csv')['napetsot']*1000

def fit_kerr(x, a, b, c):
    return a*(np.sin(b * x**2 + c / 2))**2

popt1, pcov = curve_fit(fit_kerr, kerr['V'], kerr['P'], [8.5e-4, 5.5e-6, -1])

perr = np.sqrt(np.diag(pcov))

print('kerr:')
print(popt1, perr)

def fit_kerr2(x, a, b, c, d):
    return a*(np.sin(b * (x-d)**2 + c / 2))**2

popt, pcov = curve_fit(fit_kerr2, kerr['V'], kerr['P'], [8.5e-4, 5.5e-6, -1, 220])

perr = np.sqrt(np.diag(pcov))

print('kerr2:')
print(popt, perr)

v = np.linspace(30, 1000, 1000)

plt.plot(kerr['V'], kerr['P'], 'o', ms=2, label='meritve')
plt.plot(v, fit_kerr(v, *popt1), label='fit 1')
plt.plot(v, fit_kerr2(v, *popt), label='fit 2')
plt.title('Odvisnost toka od napetosti na keramiki')
plt.grid()
plt.xlabel('Napetost [$V$]')
plt.ylabel('Intenziteta [relativne enote]')
plt.legend(loc='upper left')
plt.savefig('ElOpt/porocilo/kerr.pdf', dpi=1024)
plt.clf()

### tekoči kristal + poli

tk_poli = pd.read_csv('ElOpt/data/ruj-data/elopt_4_1nal.txt', sep='\t', names=['kot', 'I'])
tk_poli['kot'] = 90 - 5* tk_poli['kot']

def fit_tk1(x,i0, i1, d):
    return i0 + i1 * (np.sin((np.pi / 180) * x + d))**2

popt, pcov = curve_fit(fit_tk1, tk_poli['kot'], tk_poli['I'], [8e-7, 7e-5, -0.7])

perr = np.sqrt(np.diag(pcov))

print('tk poli:')
print(popt, perr)

plt.plot(tk_poli['kot'], tk_poli['I'], 'o', ms=2, label='meritve')
plt.plot(kot, fit_tk1(kot, *popt), label='fit')
plt.title('Odvisnost toka od zasuka polarizatorja')
plt.grid()
plt.xlabel('kot [$^{\\circ}$]')
plt.ylabel('Intenziteta [relativne enote]')
plt.legend(loc='upper left')
plt.savefig('ElOpt/porocilo/tkpoli.pdf', dpi=1024)
plt.clf()

### tekoči kristal - balerina

tk = pd.read_csv('ElOpt/data/ruj-data/elopt_4_2nal.txt', sep='\t', names=['kot', 'I'])
tk['kot'] = -110 + 5*tk['kot']
tk = tk.iloc[2:-2]

p = 1.532
v = 1.706
kot = np.linspace(-100, 100, 1000)

def tk_fit(x, i0, i1, d):
    return i0 + i1 * (np.sin(d * (np.sqrt(p**2 - np.sin((np.pi / 180)*x)**2) - np.sqrt(v**2 - np.sin((np.pi / 180)*x)**2))))**2

popt, pcov = curve_fit(tk_fit, tk['kot'], tk['I'], [1.2e-6, -4e-4, 16])

perr = np.sqrt(np.diag(pcov))

print('tk balerina:')
print(popt, perr)


plt.plot(tk['kot'], tk['I'], 'o', ms=2, label='meritve')
plt.plot(kot, tk_fit(kot, *popt), label='fit')
plt.title('Odvisnost toka od zasuka tekočega kristala')
plt.grid()
plt.legend(loc='upper left')
plt.xlabel('kot [$^{\\circ}$]')
plt.ylabel('Intenziteta [relativne enote]')
plt.savefig('ElOpt/porocilo/tk-balerina.pdf', dpi=1024)
plt.clf()

### izračun ekscentričnosti:

p0 = ufloat(1.77e-6, 0.03e-6)
p1 = ufloat(-8.1e-7, 0.4e-7)

e = umath.sqrt(1-p0**2/(p0-p1)**2)
print(e)