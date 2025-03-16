import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import scienceplots

#plt.style.use('science') #odkomentiraj ko shranjuješ sliko, drgač pa ne ker kr upočasni zadeve

files = glob.glob("UltZvo/data/*.csv")
files.sort(key=str.lower)
print(files)


file = files[2] #spreminjaj indeks za obdelavo posamezne meritve

print(file)

data = pd.read_csv(file, skiprows=12)
#print(data)
plt.plot(data.iloc[:, 0], data.iloc[:, 1])

# plt.title('Vsi skupaj')   #odkomentiraj za shranjevanje slike
# plt.xlabel('$U_1 [V]$')
# plt.ylabel('$I_2 [nA]$')
# plt.legend()
# plt.grid()
# plt.savefig('FraHer/porocilo/vsi.pdf', dpi=1024)

plt.grid()
plt.show()

plt.clf()