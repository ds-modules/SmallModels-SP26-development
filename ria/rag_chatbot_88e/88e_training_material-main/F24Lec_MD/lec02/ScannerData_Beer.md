---
title: "ScannerData_Beer"
type: lecture-notebook
week: 2
source_path: "/Users/ericvandusen/Documents/Data88E-ForTraining/F24Lec_NBs/lec02/ScannerData_Beer.ipynb"
---

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
%matplotlib inline
```

```python
try:
  import gdown
except:
  !pip install gdown
  import gdown
```

```python
#This file is too big and appears to crash the Datahub - could potentially work on a local machine with more RAM
#!  gdown 1z0XeoE6PYPOUqzzKLagvmiFohdOF3CLJ
# n_wber.csv = 140M
```

This is a smaller version of the dataset with half of the data ( approx 2m rows) 

The following command pulls the dataset over from Google Drive

```python
#https://drive.google.com/file/d/1qZpdbEpvvWQJlZKpzu8rwd53JrrufzdW/view?usp=sharing
!  gdown 1qZpdbEpvvWQJlZKpzu8rwd53JrrufzdW
# s_wber.csv = 70M
```

# A Quick Look on the Inverse Demand Curves for Beer

Start off by pulling in open source data on beer sales from [The University of Chicago Booth School of Business](https://www.chicagobooth.edu/research/kilts/datasets/dominicks).

```python
df = pd.read_csv('s_wber.csv')
df
```

Can you tell what each instance (row) and the features (columns) represent?

```python
df.describe()
```

One way that I like to approach a dataset where I don't know the specific details of it is to first summarize the whole table, then delve deeper into each feature. Here's a quick example of my approach:

```python
len(df['UPC'].unique()) # Number of Unique beers?
```

```python
len(df['STORE'].unique()) # Number of unique stores
```

```python
len(df['WEEK'].unique()) # Number of unique weeks
```

```python
df['PRICE'].mean() # Average beer price
```

```python
df['PRICE'].max() # most expensive beer
```

```python
df['PRICE'].min() # FREE BEER?!
```

```python
df['QTY'].mean() # avg beers bought ?
```

```python
df['MOVE'].mean() # avg beers bought ?
```

Can you tell what this plot is showing below? I'm not sure if I can!

```python
df[['MOVE','PRICE']].plot();
```

```python
df['QUANTITY'] = df['MOVE'] # Interestingly enough, quantity is not denoted as QTY, by 'MOVE'.
```

By now, we should be aware that we're looking at a dataset of beer sales, where the respective price and quantities for each transaction is represented. Let's filter out all the free beer - although it would be very nice to keep that!

```python
g = df[['QUANTITY','PRICE']]
g = g[g['PRICE']>0]
g
```

To create the demand curve itself, we need to remember that we're looking for quantity demanded at each given price. Hence, we group by price and 'ask' for the sum at the given price. Then, flip that around (just trick), and do the cumulative sum (since a demand curve is cumulative), and then flip it around one last time (since we're looking at the inverse demand curve).

```python
demand = np.flip(np.cumsum(np.flip(g.groupby('PRICE')['QUANTITY'].sum())))
demand = demand.to_frame().reset_index()
demand
```

Now, let's use Plotly Express' [Scatterplot function](https://plotly.com/python-api-reference/generated/plotly.express.scatter.html) to visualize the inverse demand curve for all beers!

```python
px.scatter(demand,x='QUANTITY', y='PRICE', trendline='ols', title='Inverse Demand Curve for all Beers')
```

This is pretty cool! Hover over the line to see the Ordinary Least Squares approximation of the inverse demand curve. What does it tell you?

Now, do you expect the price elasticity of demand to differ for different prices? Yes! Usually, a more expensive good (luxury beers?) tend to have a higher PED. Could we visualize this?

```python
demand.describe()
```

Let's plot multiple demand curves for different price segments of beer. We could start with all beers above the mean. Let's call them expensive.

```python
demand['EXPENSIVE'] = demand['PRICE']>demand['PRICE'].mean()
demand.head(5)
```

Before you plot, think about how this curve might differ from the previous one. Then, check if your intuition was right!

```python
px.scatter(demand,x='QUANTITY', y='PRICE', trendline='ols', color='EXPENSIVE',title='Inverse Demand Curve for Expensive and Cheaper Beer')
```

Did your economic intuition help you? Now, what's happening with the really expensive beer?

```python
demand['REALLY EXPENSIVE'] = demand['PRICE']>10.5 #75th Percentile of Price
demand.head(5)
```

```python
px.scatter(demand,x='QUANTITY', y='PRICE', trendline='ols', color='REALLY EXPENSIVE',title='Inverse Demand Curve for Really Expensive Beer')
```

And the really, really expensive beer?

```python
demand['REALLY, REALLY EXPENSIVE'] = demand['PRICE']>15
demand.head(5)
```

```python
px.scatter(demand,x='QUANTITY', y='PRICE', trendline='ols', color='REALLY, REALLY EXPENSIVE',title='Inverse Demand Curve for Really, Really Expensive Beer is almost Vertical!')
```

This notebook should have given you the data science skills to plot up simple, but powerful inverse demand curves. It should have also gotten you thinking about how demand curves differ for different price segments for the same goods.

Made by Peter F. Grinde-Hollevik.

```python

```

