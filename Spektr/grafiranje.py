import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import scienceplots
from scipy.optimize import curve_fit

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

x = np.linspace(434.2, 656.9, 1000)

# plt.errorbar(cali1['val'], cali1['kot'], yerr=cali1['kot err'], fmt='o', label='Data')
# plt.plot(x, fit(x, *popt), 'r-', label='Fit')
# plt.legend()
# plt.show()