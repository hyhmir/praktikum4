#plot on the same graph all the csv files from the data folder using pandas and matplotlib. 
# for each file plot the second and third column, from 13th line onwards  

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

files = glob.glob("data/*.csv")
for file in files:
    print(file)
    data = pd.read_csv(file, skiprows=12)
    print(data)
    plt.plot(data.iloc[:, 1], data.iloc[:, 2])
plt.show()

