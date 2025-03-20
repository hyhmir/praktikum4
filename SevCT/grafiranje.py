import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import scienceplots

plt.style.use('science')

files = glob.glob("SevCT/data/*.csv")

podatki = files[1]

data = pd.read_csv(podatki)

# print(f'povprečje: {data['moč (ščit)'].mean()}, sd: {data["moč (ščit)"].std()}') # za odštet ozadje


#računanje celotne izsevane moči

data['$P_{i,b}$'] = (data['moč (brez)'] - data['moč (ščit)'].mean()).round(5)
data['$P_{i,si}$'] = (data['moč (silicij)'] - data['moč (ščit)'].mean()).round(5)
data['$P_{c,b}$'] = (data['$P_{i,b}$'] * 15402.6535).round(5)
data['$P_{c,si}$'] = (data['$P_{i,si}$'] * 15402.6535).round(5)


data[['$P_{i,b}$', '$P_{i,si}$', '$P_{c,b}$', '$P_{c,si}$']].to_csv('SevCT/data/tabelca-moci.csv', index=False)


#grafiranje izsevana proti električna moč
x = np.linspace(0, 40, 100)

plt.errorbar(data['tok'] * data['napetost'], data['$P_{c,b}$'], yerr=0.05 * data['$P_{c,b}$'], fmt='o', label='$P_{c,b}$ proti P_{el}')
plt.plot(x,x)
plt.grid()
plt.legend()
plt.title('Izsevana moč v odvisnosti električne moči')
plt.xlabel('$P_{el}$')
plt.ylabel('$P_{izs}$')
plt.savefig('SevCT/porocilo/izkoristek.pdf', dpi=1024)

plt.clf()

#računanje in grafiranje upora v odvisnosti od temperature

plt.errorbar(295.4, 120, xerr=0.5, yerr=1, fmt='o') #sobna temp

data['T'] = 2700 * (data['tok'] * data['napetost'] / 30) ** (1/4)
data['errT'] = data['T'] * 0.03
data['R'] = data['napetost'] / data['tok']
data['errR'] = (0.1 * data['tok'] + 1e-3 * data['napetost']) / data['tok'] ** 2


coeff = np.polyfit(data['T'], data['R'], 1) #fitting
m, b = coeff
T = np.linspace(294, 3000, 200)
R = m * T + b
print(coeff)

plt.errorbar(data['T'], data['R'], xerr=data['errT'], yerr=data['errR'], fmt='o', label='R od T') #graphing
plt.plot(T, R, label='fit')
plt.grid()
plt.legend()
plt.title('Upor žarilne nitke v odvisnosti od temperature')
plt.xlabel('$T\ [K]$')
plt.ylabel('$R\ [\Omega]$')
plt.savefig('SevCT/porocilo/upor.pdf', dpi=1024)

