import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None) # reading iris.csv
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'] # organizing nomenclature
df.replace({'Iris-setosa': 'setosa'}, regex=True, inplace=True)
df.replace({'Iris-versicolor': 'versicolor'}, regex=True, inplace=True)
df.replace({'Iris-virginica': 'virginica'}, regex=True, inplace=True)


# 1 - Summary

save_exit = sys.stdout # outputs a summary of each variable to a single text file
file = open('Summary.txt', 'w')
sys.stdout = file
w = df.head(10)
print(w)
print(df.info())
sys.stdout = save_exit
file.close()


# 2 - Histogram

df.groupby('species')['petal_length'].sum().plot(kind='barh', legend='Histogram of Iris Petal Length')
plt.xlabel('length')
plt.ylabel('frequency')
plt.savefig('Iris_histogram.png', transparent=True) # saves a histogram of each variable to png files
plt.show()
plt.close()


# 3 - Scatter - matrix plots shows the data and its distribution (https://pandas.pydata.org/docs/reference/api/pandas.plotting.scatter_matrix.html)

new = df[['species','petal_length', 'sepal_length', 'petal_width', 'sepal_width']] # outputs a scatter plot of each pair of variables
sns.set(style='ticks', color_codes=True)
g = sns.pairplot(new, hue='species', palette='Spectral')
plt.savefig('Iris_scatter.png', transparent=True)
plt.show()

