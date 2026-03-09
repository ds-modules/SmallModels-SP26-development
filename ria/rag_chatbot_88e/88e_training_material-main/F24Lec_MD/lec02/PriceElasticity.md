---
title: "PriceElasticity"
type: lecture-notebook
week: 2
source_path: "/Users/ericvandusen/Documents/Data88E-ForTraining/F24Lec_NBs/lec02/PriceElasticity.ipynb"
---

## Price Elasticity of Demand

```python
def calculate_ped(initial_quantity, new_quantity, initial_price, new_price):
    # Calculate the percentage change in quantity demanded
    percent_change_in_quantity = ((new_quantity - initial_quantity) / initial_quantity) * 100
    
    # Calculate the percentage change in price
    percent_change_in_price = ((new_price - initial_price) / initial_price) * 100
    
    # Calculate the price elasticity of demand (PED)
    ped = percent_change_in_quantity / percent_change_in_price
    
    return ped
```

## Let's start with the Textbook Example
https://data88e.org/textbook/content/01-demand/04-elasticity.html

![](PED-txt.png)

### Example 1

```python

initial_quantity = 5
new_quantity = 4
initial_price = 5
new_price = 6

ped = calculate_ped(initial_quantity, new_quantity, initial_price, new_price)
print(f"The price elasticity of demand is {ped:.2f}")
```

### Example 2

```python
initial_quantity = 4
new_quantity = 5
initial_price = 6
new_price = 5

ped = calculate_ped(initial_quantity, new_quantity, initial_price, new_price)
print(f"The price elasticity of demand is {ped:.2f}")
```

## How about the point slope formula for eslasticity of demand?

Slope = -1 along the line

```python
slope=-1
```

### Example 1  - point slope

```python
initial_quantity = 5
initial_price = 5
ped = slope * (initial_price/initial_quantity)
print(f"The price elasticity of demand is {ped:.2f}")
```

### Example 2  - point slope

```python
initial_quantity = 4
initial_price = 6
ped = slope * (initial_price/initial_quantity)
print(f"The price elasticity of demand is {ped:.2f}")
```

```python

```

