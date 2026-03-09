---
title: "lec04-CSfromSurvey"
type: lecture-notebook
week: 4
source_path: "/Users/ericvandusen/Documents/Data88E-ForTraining/F24Lec_NBs/lec04/lec04-CSfromSurvey.ipynb"
---

<table style="width: 100%;">
    <tr style="background-color: transparent;"><td>
        <img src="https://data-88e.github.io/assets/images/blue_text.png" width="250px" style="margin-left: 0;" />
    </td><td>
        <p style="text-align: right; font-size: 10pt;"><strong>Economic Models</strong>, Fall 2024<br>
            Dr. Eric Van Dusen <br>
        Kidong Kim</p></td></tr>
</table>

# Lecture 4: Demand Survey and Surplus #

The idea for this demo is to use the student's demand curve to motivate the concept of surplus.

```python
from datascience import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
%matplotlib inline

import sympy
solve = lambda x,y: sympy.solve(x-y)[0] if len(sympy.solve(x-y))==1 else "Not Single Solution"

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import display
import warnings
warnings.filterwarnings("ignore")
```

## Section 1: Market Demand and Consumer Surplus

Let's start off with the demand from a student demand survey:
 - We had 4 "goods" and a range of prices available for each good
 - Students made bids on their willingness to pay for each of the 4 goods
 - The dataset for Fall 2022 has ~100 observations
 - This dataset has been exported and we read it in below:

```python
# sheet_id = "1jzgX74fgWo91Dyv7SbD4AmSFvUN5APc79BaOENpsyv8"
# sheet_name = "Form1"
# url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

sheet_id = "1jzgX74fgWo91Dyv7SbD4AmSFvUN5APc79BaOENpsyv8"
sheet_name = "Form1"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"


df_demand=pd.read_csv(url)

DemandTable = Table.from_df(df_demand)
DemandTable = DemandTable.drop('Timestamp')
DemandTable
```

```python
for i in DemandTable.labels:
    DemandTable.ihist(i, bins=7);
```

**Let's focus on the burritos first.** How many people are willing to buying a gourment burrito at any given price?   
We can assume that a person would be willing to buy the good at a price less than their bid price.

```python
BurritosTable = DemandTable.select('Burrito')
BurritosTable
```

```python
# Count how many people are in each answer pool
BurritosTable.group("Burrito")
```

```python
# Create a bar plot
table = BurritosTable.group("Burrito")

def plot_histogram(data, bins, title="Title", x_label = "Price", y_label = "Count"):
    plt.bar(bins, data, edgecolor="brown", align="center", width = 2)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    return 

burrito_bins = table.column(0) # Select column using method call
burrito_data = table['count'] # Select column using indexing
burrito_title = "Demand of Burritos according to different prices"

plot_histogram(burrito_data, burrito_bins, burrito_title)
```

In the visualization above, the height of each bar isn't quite right - someone who is willing to pay \\$10 for a burrito will also pay \\$2.5 for the same burrito.

```python
Qdemand = np.flip(np.cumsum(np.flip(BurritosTable.group("Burrito").column("count"))))
Qdemand
```

```python
DemandBurr= Table().with_columns([
    'priceBurr', [2.5, 5, 7.5, 10.00, 12.5, 15,17.5, 20], # those are the prices
    'Qdemand', Qdemand
])
DemandBurr
```

```python
burrito_Qdemand_bins = DemandBurr.column('priceBurr') # Select column using method call
burrito_Qdemand_data = DemandBurr['Qdemand'] # Select column using indexing
burrito_Qdemand_title = "Quantity demanded of Burritos with different prices"
Qdemand_x_label = "Price of Burrito"
Qdemand_y_label = "Quantity demanded"

plot_histogram(burrito_Qdemand_data, burrito_Qdemand_bins, burrito_Qdemand_title, Qdemand_x_label, Qdemand_y_label)
```

### Let's take a look at this table and think about the Consumer Surplus

First, let's sort the table from the most expensive burritos to the least expensive ones. 

Then, if the price is \$10, how many people are willing to pay more than the price? These people would be getting a **surplus** by only having to pay a cheaper price than the one they would be willing to pay.

```python
DemandBurr.sort("priceBurr", descending = True)
```

It looks like 
- 52 people would have been willing to pay up to \\$12.5
- 25 people would have been willing to pay up to \\$15.0
- 8 person would have been willing to pay up to \\$17.5
- 4 people would have been willing to pay up to \\$20.0

Let's add up these values

```python
CS_counting = 52*(12.5-10)+25*(15-10)+8*(17.5-10)+4*(20-10)
print('The consumer surplus from counting consumers is', CS_counting)
```

```python
# calculate the total consumer surplus given a demand table and the price of the good
def consumer_surplus(demand_table, price):
    
    # only people with a willingness to pay higher than the market price will buy the good
    demand_table_in_market = demand_table.where(0, are.above_or_equal_to(price))
    
    cs = (demand_table_in_market.column(0) - price) * demand_table_in_market.column(1)
    total_cs = sum(cs)
    
    return total_cs
```

```python
cs_burrito = consumer_surplus(DemandBurr, 10)
print('The consumer surplus from counting consumers is', cs_burrito)
```

How can we visualize the consumer surplus on the demand and supply diagram? We'll start by creating a demand curve first like before.

```python
DemandBurr.scatter("Qdemand", "priceBurr")
plt.xlabel('Quantity')
plt.ylabel('Price')
plt.title('Demand for Gourmet Burrito');
```

```python
DemandBurr.plot("Qdemand", "priceBurr", linewidth= 3)
plt.xlabel('Quantity')
plt.ylabel('Price')
plt.title('Demand for Gourmet Burrito');
```

Now that we have a demand curve - Let's create for a model that makes a linear approximation like we did in lecture 2.

```python
DemandGM = np.polyfit(DemandBurr.column("Qdemand"), DemandBurr.column("priceBurr"),1)
DemandGM
```

```python
burr_slope = DemandGM.item(0)
burr_slope
```

```python
burr_intercept = DemandGM.item(1)
burr_intercept
```

```python
# plot the actual demand curve
DemandBurr.plot("Qdemand", "priceBurr", linewidth=3)

# plot the linear approximation
burr_quantities = np.arange(0,120,0.01)
burr_prices = burr_slope * burr_quantities + burr_intercept
plt.plot(burr_quantities, burr_prices, linewidth=3)

plt.xlabel('Quantity')
plt.ylabel('Price')
plt.title('Demand for Burrito');
```

How many people does the model think would buy burritos when its price is \\$10? Let's answer this question using sympy.

```python
# Set up the demand curve expression 
Q = sympy.Symbol("Q")
demand = burr_slope * Q + burr_intercept

# Solve for Q_star when price is 10
Q_Star = solve(demand, 10)
Q_Star
```

Now we will visualize the consumer surplus of the burrito market.

```python
DemandBurr.plot("Qdemand", "priceBurr", linewidth= 3) #Black : Demand for buritto

plt.plot(np.arange(0,82,0.01), burr_slope * np.arange(0,82,0.01) + burr_intercept, linewidth= 3) #Blue : 

price = 10
plt.plot([0,Q_Star],[price, price], color = 'r', linewidth= 3) #Red : Price

triangle1 = patches.Polygon([[0,10],[Q_Star,10],[0,burr_intercept]],True,color="green") #Consumer surplus
currentAxis = plt.gca()
currentAxis.add_patch(triangle1)

plt.xlabel('Quantity')
plt.ylabel('Price')
plt.title('Demand for Burrito')

burr_price = 10

# Code for Slope and Intercept - What are the slope and intercept of the fit line
std_units = lambda a: (a - np.mean(a)) / np.std(a)
corr = lambda x, y: np.mean(std_units(x) * std_units(y))
slope = lambda x, y: corr(x, y) * np.std(y) / np.std(x)
intercept = lambda x, y: np.mean(y) - slope(x, y) * np.mean(x)



# Sums up the surplus at the give price
def surplus(bins, data, price):
    #Finding the quatity at the point where red line indicating surplus and blue line indicating the relationship btw quantiy and price.
    slope_sur = slope(data, bins)
    intercept_sur = intercept(data, bins)
    Q = sympy.Symbol("Q")
    demand = slope_sur * Q + intercept_sur
    Q_Star = solve(demand, price)
    
    #Sums up the surplus and print it out.
    total_surplus = .5 * Q_Star * (intercept_sur - price)
    return total_surplus

agg_surplus = surplus(DemandBurr["priceBurr"], DemandBurr["Qdemand"], burr_price)
print("Consumer surplus is equal to green triangle: " + str(0.5 * (burr_intercept - burr_price) * Q_Star))
```

###  Let's try again for Greek Theater Tickets

```python
GreekTixTable = DemandTable.select('GreekTix')
GreekTixTable
```

```python
# apply the same trick to obtain demand at each price
Qdemand = np.flip(np.cumsum(np.flip(GreekTixTable.group("GreekTix").column("count"))))
```

```python
DemandGreekTix = Table().with_columns([
    'priceTix', [25, 50, 75, 100, 125, 150, 175, 200],
    'Qdemand', Qdemand
])
DemandGreekTix
```

```python
tix_slope = slope(DemandGreekTix["Qdemand"], DemandGreekTix["priceTix"])
tix_intercept = intercept(DemandGreekTix["Qdemand"], DemandGreekTix["priceTix"])
tix_slope, tix_intercept
```

How many people does the Model think would buy at \\$100? Let's again use sympy.

```python
solve = lambda x,y: sympy.solve(x-y)[0] if len(sympy.solve(x-y))==1 else "Not Single Solution"
Q = sympy.Symbol("Q")
demand = tix_slope * Q + tix_intercept

Q_Star = solve(demand, 100)
Q_Star
```

Visualize the consumer surplus for Greek Theater tickets!

```python
DemandGreekTix.plot("Qdemand", "priceTix", linewidth=3, zorder=20) #Black : Demand for Greek Theater

triangle1 = patches.Polygon([[0,100],[Q_Star,100],[0,tix_intercept]], True, color="green", zorder=1)
currentAxis = plt.gca()
currentAxis.add_patch(triangle1)

plt.plot(np.arange(0,82,0.01), tix_slope * np.arange(0,82,0.01) + tix_intercept, linewidth= 3, zorder=5) #Blue : Demand

price = 100

# This line to interactive version
plt.plot([0,Q_Star],[price]*2, color = 'r', linewidth= 3, zorder = 10) #Red : Price

plt.xlabel('Quantity')
plt.ylabel('Price')
plt.title('Demand for Greek Theater Tickets');

agg_surplus = surplus(DemandGreekTix["priceTix"], DemandGreekTix["Qdemand"], price)
print("Consumer surplus is equal to green triangle: " + str(agg_surplus))
```

