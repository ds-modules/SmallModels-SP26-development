---
title: "demand-curve-Fa24"
type: lecture-notebook
week: 2
source_path: "/Users/ericvandusen/Documents/Data88E-ForTraining/F24Lec_NBs/lec02/demand-curve-Fa24.ipynb"
---

# Creating a Demand Curve

```python
import pandas as pd
import os
import json
import numpy as np
from datascience import *
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
%matplotlib inline
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
```

We will create a few demand curves based on the class. Start by filling in the form at https://forms.gle/KhCjC4nwwhtRDjSJ9 or https://tinyurl.com/data-88e-demand!

```python
sheet_id = "1jzgX74fgWo91Dyv7SbD4AmSFvUN5APc79BaOENpsyv8"
sheet_name = "Form1"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
```

```python
pd.read_csv(url)
```

```python
df_demand=pd.read_csv(url)
```

```python
demand_table = Table.from_df(df_demand)
demand_table = demand_table.drop('Timestamp')
demand_table
```

Let's try graphing all our different responses!

```python
for i in demand_table.labels:
    demand_table.hist(i);
```

Let's start by looking at the demand for masks. How many people would buy masks at a given price? Let's assume that a person would be willing to buy the good at a price less than their bid price.

```python
# This is a column of bid values for masks that you've all inputted. 
masks = demand_table.select('Masks')
masks
```

```python
# This cell does some python magic. You do not need to worry about what's going on. 
prices = pd.DataFrame({'price':[0.25, 0.5, 0.75, 1.00, 1.25, 1.5,1.75,2]})
MasksByPrice = masks.group("Masks")
mbp = MasksByPrice.to_df()
mask = (
    prices
    .merge(mbp, left_on='price', how='left', right_on='Masks')
    .fillna(0).drop('Masks', axis=1)
)
masks_table = Table.from_df(mask)
Q_demand = np.flip(np.cumsum(np.flip(masks_table.group("price", sum).column(1))))
masks_demand = Table().with_columns(
    'price', prices.price, 
    'quantity', Q_demand
)
masks_demand
```

```python
# Let's graph our results
masks_demand.scatter("quantity", "price")
plt.xlabel('Quantity')
plt.ylabel('Price')
plt.title('Demand for a pack of surgical masks');
```

Now let's find the slope and intercept of the line of best fit. The cell below defines some functions that you'll learn about in the later portions of Data 8.

```python
std_units = lambda a: (a - np.mean(a)) / np.std(a)
corr = lambda x, y: np.mean(std_units(x) * std_units(y))
slope = lambda x, y: corr(x, y) * np.std(y) / np.std(x)
intercept = lambda x, y: np.mean(y) - slope(x, y) * np.mean(x)
```

```python
slope(masks_demand["quantity"], masks_demand["price"])
```

```python
intercept(masks_demand["quantity"], masks_demand["price"])
```

We can use the same code as above to create demand curves for our other products as well!

```python
#Gourmet Burrito
prices_burrito = pd.DataFrame({'price':[2.50, 5, 7.50, 10, 12.5, 15,17.5,20]})

burritos = demand_table.select('Burrito')
burritosByPrice = burritos.group("Burrito")
bbp = burritosByPrice.to_df()
gb = (
    prices_burrito
    .merge(bbp, left_on='price', how='left', right_on='Burrito')
    .fillna(0).drop('Burrito', axis=1)
)

burritos_table = Table.from_df(gb)
Q_demand_burrito = np.flip(np.cumsum(np.flip(burritos_table.group("price", sum).column(1))))

gb_demand = Table().with_columns(
    'price', prices_burrito.price, 
    'quantity', Q_demand_burrito
)

burrito_slope = slope(gb_demand["quantity"], gb_demand["price"])
burrito_intercept = intercept(gb_demand["quantity"], gb_demand["price"])
print("Slope: " + str(burrito_slope))
print("Intercept: " +  str(burrito_intercept))
```

```python
#Greek Theatre Tickets
prices_tickets = pd.DataFrame({'price':[25, 50, 75, 100, 125, 150,175,200]})

tickets = demand_table.select('GreekTix')
ticketsByPrice = tickets.group("GreekTix")
tbp = ticketsByPrice.to_df()
gt = (
    prices_tickets
    .merge(tbp, left_on='price', how='left', right_on='GreekTix')
    .fillna(0).drop('GreekTix', axis=1)
)

tickets_table = Table.from_df(gt)
Q_demand_tickets = np.flip(np.cumsum(np.flip(tickets_table.group("price", sum).column(1))))

gt_demand = Table().with_columns(
    'price', prices_tickets.price, 
    'quantity', Q_demand_tickets
)

tickets_slope = slope(gt_demand["quantity"], gt_demand["price"])
tickets_intercept = intercept(gt_demand["quantity"], gt_demand["price"])
print("Slope: " + str(tickets_slope))
print("Intercept: " +  str(tickets_intercept))
```

```python
#Iphone 14
prices_iphone = pd.DataFrame({'price':[250, 500, 750, 1000, 1250, 1500,1750,2000, 2250, 2500, 2750, 3000]})

iphones = demand_table.select('iPhone')
iphonesByPrice = iphones.group("iPhone")
ibp = iphonesByPrice.to_df()
iphone14 = (
    prices_iphone
    .merge(ibp, left_on='price', how='left', right_on="iPhone")
    .fillna(0).drop("iPhone", axis=1)
)

iphones_table = Table.from_df(iphone14)
Q_demand_iphones = np.flip(np.cumsum(np.flip(iphones_table.group("price", sum).column(1))))

iphone14_demand = Table().with_columns(
    'price', prices_iphone.price, 
    'quantity', Q_demand_iphones
)

iphones_slope = slope(iphone14_demand["quantity"], iphone14_demand["price"])
iphones_intercept = intercept(iphone14_demand["quantity"], iphone14_demand["price"])
print("Slope: " + str(iphones_slope))
print("Intercept: " +  str(iphones_intercept))
```

Comparing the demand curves for our four products, what similarities or differences do you notice? In particular, think about what the slopes of the curves might reveal to us about consumer preferences.

```python
masks_demand.scatter("quantity", "price")
plt.xlabel('Quantity')
plt.ylabel('Price')
plt.title('Demand for a pack of surgical masks');

gb_demand.scatter("quantity", "price")
plt.xlabel('Quantity')
plt.ylabel('Price')
plt.title('Demand for Gourmet Burritos');

gt_demand.scatter("quantity", "price")
plt.xlabel('Quantity')
plt.ylabel('Price')
plt.title('Demand for Greek Theatre Tickets');

iphone14_demand.scatter("quantity", "price")
plt.xlabel('Quantity')
plt.ylabel('Price')
plt.title('Demand for iPhone14');
```



