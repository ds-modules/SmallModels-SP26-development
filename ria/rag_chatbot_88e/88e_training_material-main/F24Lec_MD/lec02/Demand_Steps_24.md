---
title: "Demand_Steps_24"
type: lecture-notebook
week: 2
source_path: "/Users/ericvandusen/Documents/Data88E-ForTraining/F24Lec_NBs/lec02/Demand_Steps_24.ipynb"
---

## Demand Curve step by step


We will create a few demand curves based on the class. Start by filling in the form at 
 - https://forms.gle/uxgsxtnedqeANshCA     or 
 -  https://tinyurl.com/data88fa24demand

```python
import pandas as pd
from datascience import *
import numpy as np
%matplotlib inline
```

### Find the Sheet ID in the URL of the Google Sheet!

Take a look at the data 
https://docs.google.com/spreadsheets/d/1jp-XrFPk0eUNDUVWGa7Rmw9b0P8_jobTG0oLpvcHB9s/edit?resourcekey=&gid=418675525#gid=418675525

```python
sheet_id = "1jp-XrFPk0eUNDUVWGa7Rmw9b0P8_jobTG0oLpvcHB9s"
sheet_name = "Form1"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
```

Read it into a datascience table

```python
demand_table = Table.read_table(url)
demand_table
```

```python
demand_table.ihist("Coffee",bins=7)
```

```python
demand_table.ihist("Burrito",bins=7)
```

## Gonna roll with Burrito for this example

Step 1 - Lets pull out just Burritos 

This is a table with just Burrito prices that people are willing to pay ( bids)

```python
Burrito = demand_table.select("Burrito")
Burrito
```

Step 2 - Let's count the number at each price

And sort the table so that it is descending from high to low price

```python
# count the number at each price
Burrito_counts = demand_table.group("Burrito")
Burrito_counts = Burrito_counts.sort('Burrito', descending=True)

Burrito_counts
```

Step 3 - Let's pull out those counts

```python
counts = Burrito_counts.column("count")
print(counts)
```

Step 4 - use a numpy command called cumulative sum to get the number of people who will buy at each price

```python
cumulative_counts = counts.cumsum()
cumulative_counts
```

Step 5 - make an array of the prices of the burritos in descending order

```python
prices = make_array(20,17.5,15,12.5,10,7.5,5,2.5)
prices
```

Step 6 -  make a table with the prices and the cumulative counts

```python
demand_curve = Table().with_columns("Price", prices, "Cumulative Count", cumulative_counts)
demand_curve
```

```python
demand_curve.iscatter("Cumulative Count","Price")
```

```python
demand_curve.iscatter("Cumulative Count","Price", fit_line=True)
```

```python
# fit a line to the data using numpy        
m, b = np.polyfit(cumulative_counts,prices,  1)
print(m, b)
```

```python
# add a new column to the table with the log of price
demand_curve = demand_curve.with_column("Log Price", np.log(prices))
demand_curve
```

```python

```

```python

```

