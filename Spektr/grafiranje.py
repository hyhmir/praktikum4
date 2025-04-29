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

files = glob.glob("Spektr/data/*.csv")
files.sort(key=str.lower)

calibration = files[0]
cali = pd.read_csv(calibration)
cali1 = pd.DataFrame({
    'kot': pd.concat([cali['Hg kot'], cali['H2 kot']], ignore_index=True),
    'val': pd.concat([cali['Hg val'], cali['H2 val']], ignore_index=True)
})
cali1["kot err"] = 0.1 * np.ones(6)


def fit(x, c1, c2, c3):
    return c1 + c2*x + c3*x**0.5

popt, pcov = curve_fit(fit, cali1["val"], cali1["kot"], sigma=cali1['kot err'], absolute_sigma=True)#deep seek cooked here

perr = np.sqrt(np.diag(pcov)) 

print("Fitted parameters with errors:")
for i, (param, err) in enumerate(zip(popt, perr)):
    print(f"Parameter {i}: {param:.3f} ± {err:.3f}")

#x = np.linspace(434.2, 656.9, 1000)

# plt.errorbar(cali['Hg val'], cali['Hg kot'], yerr=0.1, fmt='o', label='$Hg$')
# plt.errorbar(cali['H2 val'], cali['H2 kot'], yerr=0.1, fmt='o', label='$H_2$')
# plt.plot(x, fit(x, *popt), 'r-', label='Fit')
# plt.title('Umeritvena krivulja')
# plt.legend()
# plt.grid()
# plt.xlabel('$\lambda\ [nm]$')
# plt.ylabel('$\phi\ [^{\circ}]$')
# plt.savefig('Spektr/porocilo/kalibracija.pdf', dpi=1024)
# plt.clf()

# če upoštevamo napako parametrov postane napaka blizu 5 % - neuporabno
a = ufloat(108.582, 0)
b = ufloat(0.051, 0)
c = ufloat(-2.834, 0)

# a = 108.582 # za wolfram
# b = 0.051
# c = -2.834

def inv_fit(x):
    return ((-c-(c**2-4*b*(a-x))**0.5)/(2*b))**2

obdelava = pd.read_csv(files[6], names=['kot']) # ustrezno spreminjaj indeks pri obdelavi, nekje je treba tut razširit names


obdelava['uvalue'] = unumpy.uarray(obdelava['kot'], 0.1)


results = [inv_fit(x) for x in obdelava['uvalue']]

print(obdelava)

for result in results:
    print(f"Result: {result}")

results.sort(reverse=True)

display = pd.DataFrame()
display['kot'] = np.sort(np.array(obdelava['uvalue']))
display['val'] = results
print(display.to_latex(index=False))

# LED obdelava

# led = pd.read_csv(files[3])
# led['umax'] = unumpy.uarray(led['max'], 0.1)
# led['umin'] = unumpy.uarray(led['min'], 0.1)
# led['val max'] = [inv_fit(x) for x in led['umin']]
# led['val min'] = [inv_fit(x) for x in led['umax']]
# led['val mean'] = led['val max']/2 + led['val min']/2
# led['val width'] = -led['val max'] + led['val min']

# del led['max']
# del led['min']

# print(led.to_latex(index=False))

# wolfram

# wolf = pd.read_csv(files[4], names=['kot'])
# wolf['val']= [inv_fit(x) for x in wolf['kot']]
# wolf['val'] = wolf['val'].round(0)
# wolf['kot'] = wolf['kot'].round(1)
# wolf['barva'] = ['rdeča', 'rumena', 'zelena', 'modra', 'vijolična', '']
# print(wolf)
# print(wolf.to_latex(index=False))



