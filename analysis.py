import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# reading iris.csv

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print(df.head(20))

# organizing nomenclature

df.columns = ['petallength', 'petalwidth','sepallength', 'sepalwidth', 'species']

# removing the string 'Iris-' from the beginning of each type of species

df['species'] = df['species'].str.replace(r'Iris-setosa', 'setosa')
df['species'] = df['species'].str.replace(r'Iris-virginica', 'virginica')
df['species'] = df['species'].str.replace(r'Iris-versicolor', 'versicolor')

# 1 - Summary

tx = sys.stdout # outputs a summary of each variable to a single text file
openfile = open('Summary.txt', 'w')
sys.stdout = openfile
print(df.describe(include='all'))
x = np.random.randint(low=0, high=100, size=100)
m = np.mean(x)
print(m)
print(df['petallength'].corr(df['sepallength']))
print(df['petallength'].corr(df['sepalwidth']))
print(df['petallength'].corr(df['petallength']))
print(df['petallength'].corr(df['petalwidth']))
print(df['sepallength'].corr(df['petallength']))
print(df['sepallength'].corr(df['petalwidth']))
print(df['sepallength'].corr(df['sepallength']))
print(df['sepallength'].corr(df['sepalwidth']))
sys.stdout = tx
openfile.close()

# 2 - Histogram

df.groupby('species')['petallength', 'sepallength'].sum().plot(kind='barh', legend='Histogram of Iris Length')
plt.xlabel('frequency')
plt.savefig('hist_length.png', transparent=True) # saves a histogram of each variable to png files
plt.show()
plt.close()

df.groupby('species')['petalwidth', 'sepalwidth'].sum().plot(kind='barh', legend='Histogram of Iris Width')
plt.xlabel('frequency')
plt.savefig('hist_width.png', transparent=True) # saves a histogram of each variable to png files
plt.show()
plt.close()

# 3 - Scatter

new = df[['species','petallength', 'sepallength', 'petalwidth', 'sepalwidth']]
sns.set(style='ticks', color_codes=True)
g = sns.pairplot(new, hue='species', palette='Spectral')
plt.savefig('Iris_scatter.png', transparent=True)
plt.show()

