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

nj = 18.3e-6
g = 9.81
rho1 = 973
rho2 = 1.3
E = ufloat(5e4, 1e3)
pi = np.pi


data = pd.read_csv('Milik/data/data.csv')

data['v+'] = [ufloat(x*1e-6, 2e-6) for x in data['v+']]
data['v-'] = [ufloat(x*1e-6, 2e-6) for x in data['v-']]


data['r'] = unumpy.sqrt(9*nj*(data['v+'] + data['v-'])/(4*g*(rho1 - rho2)))
data['ne'] = np.abs((3*pi*data['r']*nj/E)*(data['v+']-data['v-']))

###formating data v lepe enote

data['v+'] = data['v+']*1e6
data['v-'] = data['v-']*1e6
data['r'] = data['r']*1e6
data['$n$'] = 8*[0] + 22*[1]

def safe_ufloat_divide(row):
    try:
        return row['ne'] / row['$n$']
    except (ZeroDivisionError, TypeError):
        return None

data['$e_0$'] = data.apply(safe_ufloat_divide, axis=1)

print(data.to_latex(index=False))

data['count'] = np.arange(1, len(data['ne']) + 1)
data['ne'] = unumpy.nominal_values(data['ne'])

plt.title('Kumulativna razporeditev naboja')
plt.step(data['ne'], data['count'], where='post')
plt.grid()
plt.xlabel('ne')
plt.ylabel('N')
plt.savefig('Milik/porocilo/graf.pdf', dpi=1024)
plt.clf()

# Function to calculate average of ufloat values, ignoring None
def ufloat_average(series):
    valid_values = [x for x in series if x is not None]
    if not valid_values:
        return None
    return sum(valid_values) / len(valid_values)

# Calculate average
average = ufloat_average(data['$e_0$'])
print(average)
