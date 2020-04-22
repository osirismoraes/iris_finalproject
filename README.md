# **iris_finalproject**

## Data description

Ronald Fisher, biologist and statistician in 1936 presented a set of data. 

The iris dataset contains the following data:

* 50 samples of 3 different species of iris (150 samples total)
* Measurements: sepal length, sepal width, petal length, petal width
* The format for the data: (sepal length, sepal width, petal length, petal width)

Just for reference, here are pictures of the three flowers species:

![figure1](https://user-images.githubusercontent.com/60973011/79228030-b8bf7b80-7e58-11ea-98ff-f54ad51d2731.png)
image from (https://mc.ai/visualization-and-understanding-iris-dataset/)


## Libraries Used

Importing the libaries for this project: Pandas, Numpy and Seaborn.

Pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools.

NumPy is the fundamental package for scientific computing with Python.

Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics.

## Reading the dataset

I imported iris.csv using the panda library and look at the first few lines of data

```python
 df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
 print(df.head(20))
```

## Organizing nomenclature

The columns were not named correctly, so I renamed them all.

```python
df.columns = ['petallength', 'petalwidth','sepallength', 'sepalwidth', 'species']
```

## Removing the string 'Iris-' from the beginning of each type of species

```python
df['species'] = df['species'].str.replace(r'Iris-setosa', 'setosa')
df['species'] = df['species'].str.replace(r'Iris-virginica', 'virginica')
df['species'] = df['species'].str.replace(r'Iris-versicolor', 'versicolor')
```

## Creating a summary to understand the data in an overview

This summary includes description information for all columns, the overall average and the correlation between numerical variables.

```python
tx = sys.stdout
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
```

## Creating a histogram of each variable

In this graph we can find out how the measures are distributed.

```python
df.groupby('species')['petallength', 'sepallength'].sum().plot(kind='barh', legend='Histogram of Iris Length')
plt.xlabel('frequency')
plt.savefig('hist_length.png', transparent=True) 
plt.show()
plt.close()

df.groupby('species')['petalwidth', 'sepalwidth'].sum().plot(kind='barh', legend='Histogram of Iris Width')
plt.xlabel('frequency')
plt.savefig('hist_width.png', transparent=True) 
plt.show()
plt.close()
```

## Creating a scatter plot

The graphs below show the relationship between the width and length measurements of petals and sepals of each species (see legend).

```python
new = df[['species','petallength', 'sepallength', 'petalwidth', 'sepalwidth']]
sns.set(style='ticks', color_codes=True)
g = sns.pairplot(new, hue='species', palette='Spectral')
plt.savefig('Iris_scatter.png', transparent=True)
plt.show()
```
