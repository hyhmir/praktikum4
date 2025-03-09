#plot on the same graph all the csv files from the data folder using pandas and matplotlib. 
# for each file plot the second and third column, from 13th line onwards  

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import scienceplots

plt.style.use('science')

files = glob.glob("FraHer/data/*.csv")
for file in files:
    print(file)
    data = pd.read_csv(file, skiprows=12)
    #print(data)
    plt.plot(-10 * data.iloc[:, 1], 5 * data.iloc[:, 2], label=(str(file).strip('FraHer/data/CSV').strip('.csv')) + '$^{\circ}C$')

plt.title('Vsi skupaj')
plt.xlabel('$U_1 [V]$')
plt.ylabel('$I_2 [nA]$')
plt.legend()
plt.grid()
plt.savefig('FraHer/porocilo/vsi.pdf', dpi=1024)

plt.clf()


for file in files:
    print(file)
    data = pd.read_csv(file, skiprows=12)
    #print(data)
    plt.plot(-10 * data.iloc[:, 1], 5 * data.iloc[:, 2])
    plt.title((str(file).strip('FraHer/data/CSV').strip('.csv')) + '$^{\circ}C$')
    plt.xlabel('$U_1 [V]$')
    plt.ylabel('$I_2 [nA]$')
    plt.legend()
    plt.grid()
    plt.savefig('FraHer/porocilo/' + str(file).strip('FraHer/data/CSV').strip('.csv') + '.pdf', dpi=1024)
    #plt.show() #odkomentiraj samo ko rabiš odčitat maksimume
    plt.clf()




