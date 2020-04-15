import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns


# reading iris.csv
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
df = pd.DataFrame(data)
print(df)

# outputs a summary of each variable to a single text file
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
print(df.columns)
df.replace({'Iris-setosa': 'setosa'}, regex=True, inplace=True)
df.replace({'Iris-versicolor': 'versicolor'}, regex=True, inplace=True)
df.replace({'Iris-virginica': 'virginica'}, regex=True, inplace=True)

# 1 - summary
save_exit = sys.stdout
file = open('Summary.txt', 'w')
sys.stdout = file
w = df.head(10)
print(w)
print(df.info())
sys.stdout = save_exit
file.close()

# 3 - division of `Species`
a = df[df['species'].str.contains("setosa")].groupby('species').size()
b = df[df['species'].str.contains("setosa")].groupby('species').size()
c = df[df['species'].str.contains("setosa")].groupby('species').size()
print(a)
print(b)
print(c)

# saves a histogram of each variable to png files
# 1 - histogram - the distribution of values for petal length are different for each class.
df.groupby('species')['petal_length'].sum().plot(kind='barh', legend='Histogram of Iris Petal Length')
plt.xlabel('length')
plt.ylabel('frequency')
plt.savefig('Iris_histogram.png', transparent=True)
plt.show()
plt.close()

# outputs a scatter plot of each pair of variables
# 1 - matrix plots shows the data and its distribution

new = df[['species','petal_length', 'sepal_length']]
sns.set(style='ticks', color_codes=True)
g = sns.pairplot(new, hue='species', palette='Spectral')
plt.savefig('Iris_scatter.png', transparent=True)
plt.show()

