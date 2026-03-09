---
title: "lec04-Supply-Demand-closed"
type: lecture-notebook
week: 4
source_path: "/Users/ericvandusen/Documents/Data88E-ForTraining/F24Lec_NBs/lec04/lec04-Supply-Demand-closed.ipynb"
---

<table style="width: 100%;">
    <tr style="background-color: transparent;"><td>
        <img src="https://data-88e.github.io/assets/images/blue_text.png" width="250px" style="margin-left: 0;" />
    </td><td>
        <p style="text-align: right; font-size: 10pt;"><strong>Economic Models</strong>, Fall 2024<br>
            Dr. Eric Van Dusen <br>
        </p></td></tr>
</table>

```python
import sympy
solve = lambda x,y: sympy.solve(x-y)[0] if len(sympy.solve(x-y))==1 else "Not Single Solution"
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import display
#import nbinteract as nbi
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import matplotlib
print(matplotlib.__version__)
```

# Supply, Demand, and Equilibrium

## Set-up

With this notebook, we are going to plot and solve equations, hopefully giving some more hands on exposure to the materials that you've already seen in class. 

We have already covered 
- sympy
- getting slope and intercept for deman curve
- solving for equilibrium

## Solving Equations with SymPy

In order to treat variables like the symbols you would using pen and paper, we have to declare them as such first. We are going to use the inverse forms, so our equations will both be a function of quantity. We create a variable for quantity below.

```python
Q = sympy.Symbol("Q")
```

Now when we call this variable, we can see that it is a symbolic variable, and not some other Python type.

We cn decplare an upward sloping supply curve with a name, `supply`.

```python
supply = 3*Q + 4
supply
```

We can then substitue in values for our quantity variable. We do so in the following cell. Using the method `subs`, we take the equation we already defined, `supply`, and plug in the value 3 in place of `Q`.

```python
supply.subs(Q, 3)
```

Next we'll define an inverse-demand equation.

```python
demand = -4*Q+15
demand
```

We are now able to take our supply and demand equations and find where they intersect. When we use the `solve` function, it will tell us the x-value of the point where the two lines intercept. This is the equilibrium quantity, which we will call that value `Q_star`.

```python
Q_star = solve(demand, supply) # our version of solve is simplified for single solution systems
Q_star
```

We can then substitute `Q_star` back into our original inverse-supply and inverse-demand equations to solve for our equilibrium price.

```python
demand.subs(Q, Q_star)
```

```python
supply.subs(Q, Q_star)
```

## Visualizing those supply and demand equations

```python
def plot_equation(equation, price_start, price_end, label=None):
    plot_prices = [price_start, price_end]
    plot_quantities = [equation.subs(list(equation.free_symbols)[0], c) for c in plot_prices]
    plt.plot(plot_prices, plot_quantities, label=label)
    
def plot_intercept(eq1, eq2):
    ex = sympy.solve(eq1-eq2)[0]
    why = eq1.subs(list(eq1.free_symbols)[0], ex)
    plt.scatter([ex], [why])
    return (ex, why)
    
plot_equation(demand, 0, 5)
plot_equation(supply, 0, 5)
plt.ylim(0,20)
plot_intercept(supply, demand)
```

## Modeling a Shift in Demand
We would like to be able to add in the ability to shift the demand curve, using a slider

```python

def shift_demand():
    equation = demand
    def shift_helper(shift):
        plot_equation(equation, -10, 10)
        plot_equation(supply, -10, 10)
        old = plot_intercept(equation, supply)
        print('Original Intercept:', old)
        
        if shift != 0:
            plot_equation(equation + shift, -10, 10, 'shifted')
            new = plot_intercept(equation + shift, supply)
            print('New intercept:', new)
            print('Change in Quantity:', round(float(new[0]-old[0]), 2))
            print('Change in Price:', round(float(new[1]-old[1]), 2))
        else:
            print('Nothing shifted yet, use the slider to move the line!')
        plt.xlim(0,8)
        plt.ylim(0,20)
        plt.legend()
        plt.ylabel("Price")
        plt.xlabel("Quantity")
    interact(shift_helper, shift=(-6, 12, 3))

shift_demand()
```

## Externalities
Ok lets 'shift' the analysis to 

**Marginal Private Benefit (MPB)**
The marginal private benefit (MPB) is the benefit that an individual or firm receives from consuming or producing an additional unit of a good or service. This benefit is internal to the consumer or producer and does not take into account any external costs or benefits that might affect others.

In the case of consumption, the marginal private benefit is the value a consumer places on the next unit of a good, which typically decreases as more units are consumed (due to diminishing marginal utility). This is represented by the private demand curve, which shows the relationship between the quantity demanded and the price.

For example:

If you buy a textbook, the benefit you receive from reading it is the private benefit. You might be willing to pay a certain price for the book based on how much you value the knowledge it provides.

**Marginal Social Benefit (MSB)**
The marginal social benefit (MSB) is the total benefit to society from consuming or producing an additional unit of a good or service. It includes both the marginal private benefit and any external benefits (or subtracts external costs, in the case of negative externalities).

When externalities exist, the marginal social benefit differs from the marginal private benefit. In the case of positive externalities, the MSB is greater than the MPB because the good or service provides additional benefits to third parties that are not captured by the individual consumer.

For example:

Education creates positive externalities. While the student receives private benefits from increased knowledge and skills (MPB), society as a whole also benefits from having a more educated population, leading to increased productivity, lower crime rates, and higher civic engagement. These additional benefits form the social benefit.
In a competitive market without intervention, only the private benefits are considered, leading to under-consumption of goods with positive externalities. The market does not provide enough of the good relative to the socially optimal level because individuals do not account for the benefits that spill over to others.

**Market Outcome vs. Socially Optimal Outcome**
In the presence of a positive externality:

The market outcome is determined by the intersection of the supply curve (marginal cost) and the private demand curve (MPB).
However, the socially optimal outcome occurs at the intersection of the supply curve and the social demand curve (MSB), which lies above the private demand curve due to the external benefits.

Let's do a simpler version where we just solve the following :
\begin{align*}
    \text{Private Demand (MPB)}: & \quad P = 100 - 0.3Q \\
    \text{Social Demand (MSB)}: & \quad P = (100 + \text{Externality Shift}) - 0.3Q \\
    \text{Supply (MC)}: & \quad P = 10 + 0.2Q\\
    \text {Where}: & \quad \text{Externality Shift} = +10\\
\end{align*}

```python
def private_demand(q):
    return 100 - 0.3 * q  

def social_demand(q):
    return 110 - 0.3 * q  

def supply(q):
    return 10 + 0.2 * q  

market_quantity = (100 - 10) / (0.2 + 0.3)  
optimal_quantity = (110 - 10) / (0.2 + 0.3)
```

```python

quantities = np.linspace(0, 300, 500)

#Plotting
plt.figure(figsize=(6, 6))
plt.plot(quantities, private_demand(quantities), label='Private Demand (MPB)', color='green')
plt.plot(quantities, social_demand(quantities), label='Social Demand (MSB)', color='purple', linestyle='--')
plt.plot(quantities, supply(quantities), label='Supply (MC)', color='blue')

plt.axvline(market_quantity, color='green', linestyle='--', label='Market Quantity')
plt.axvline(optimal_quantity, color='purple', linestyle='--', label='Optimal Quantity')

plt.fill_between(quantities, private_demand(quantities), social_demand(quantities), where=(quantities <= optimal_quantity), color='lightblue', alpha=0.3)

plt.title('Positive Externality: Demand Curve')
plt.xlabel('Quantity')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
```

Let's do an interactive version where we solve the following with a variable externality :
\begin{align*}
    \text{Private Demand (MPB)}: & \quad P = 100 - 0.3Q \\
    \text{Social Demand (MSB)}: & \quad P = (100 + \text{Externality Shift}) - 0.3Q \\
    \text{Supply (MC)}: & \quad P = 10 + 0.2Q
\end{align*}

```python
#Everything is wrapped in a function that the slider can interact with

def plot_externality(externality_shift):
    def social_demand(q):
        return (100 + externality_shift) - 0.3 * q  # Marginal Social Benefit (MSB)

#Hard coded values for Equilibrium from solving Demand and Supply
    market_quantity = (100 - 10) / (0.2 + 0.3) 
    optimal_quantity = ((100 + externality_shift) - 10) / (0.2 + 0.3) 
    
    market_price = private_demand(market_quantity)
    optimal_price = social_demand(optimal_quantity)

#Plotting    
    plt.figure(figsize=(8, 6))
    plt.plot(quantities, private_demand(quantities), color='green')
    plt.plot(quantities, social_demand(quantities), color='purple', linestyle='--')
    plt.plot(quantities, supply(quantities), color='blue')

    plt.axvline(market_quantity, color='green', linestyle='--', label=f'Market Quantity: {market_quantity:.2f}')
    plt.axvline(optimal_quantity, color='purple', linestyle='--', label=f'Social Quantity: {optimal_quantity:.2f}')

    plt.axhline(market_price, color='green', linestyle='--', label=f'Market Price: {market_price:.2f}')
    plt.axhline(optimal_price, color='purple', linestyle='--', label=f'Social Price: {optimal_price:.2f}')

  
    plt.fill_between(quantities, private_demand(quantities), social_demand(quantities), where=(quantities <= optimal_quantity), color='lightblue', alpha=0.3)

#Labels on demand
    plt.text(300, private_demand(300) + 2, 'Marginal Private Benefit (MPB)', color='green', fontsize=12)
    plt.text(300, social_demand(300) + 2, 'Marginal Social Benefit (MSB)', color='purple', fontsize=12)
#Labels on supply 
    plt.text(300, supply(300) + 2, 'Supply ', color='blue', fontsize=12)

  
    plt.title(f' Externality: Externality Shift = {externality_shift}')
    plt.xlabel('Quantity')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

#Slider for externality shift
interact(plot_externality, externality_shift=widgets.IntSlider(min=-20, max=20, step=1, value=10, description="Externality Shift"));
```

## Consumer and Producer Surplus

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy import symbols
import matplotlib.patches as patches

# Define the quantity symbol for sympy
Q = symbols('Q')

# Simplified Equilibrium function with a range of quantities
def Equilibrium(demandParam, supplyParam):
    # Define demand and supply equations as functions of quantity
    demandEquation = demandParam - Q  # P = demandParam - Q
    supplyEquation = (supplyParam / 10) * Q  # P = (supplyParam / 10) * Q
    
    # Convert the sympy equations to numerical functions using lambdify
    demandFunc = sympy.lambdify(Q, demandEquation, "numpy")
    supplyFunc = sympy.lambdify(Q, supplyEquation, "numpy")
    
    # Define a fixed range of quantities (from 0 to 100)
    quantities = np.linspace(0, 50, 50)
    
    # Calculate corresponding prices for each quantity for demand and supply
    demandPrices = demandFunc(quantities)  # Evaluate demand prices as numerical values
    supplyPrices = supplyFunc(quantities)  # Evaluate supply prices as numerical values

    # Find the equilibrium quantity by solving where demand equals supply
    equilibriumQ = sympy.solve(demandEquation - supplyEquation, Q)[0]
    equilibriumP = demandEquation.subs(Q, equilibriumQ)
    
    # Convert equilibrium values to float for plotting
    equilibriumQ = float(equilibriumQ)
    equilibriumP = float(equilibriumP)
    
    # Plot demand and supply curves
    plt.plot(quantities, demandPrices, label="Demand (MPB)", color="green")
    plt.plot(quantities, supplyPrices, label="Supply (MC)", color="blue")
    
    # Mark equilibrium point
    plt.plot(equilibriumQ, equilibriumP, 'ro', label=f"Equilibrium Q={round(equilibriumQ, 2)}, P={round(equilibriumP, 2)}")
    
    # Shade consumer and producer surplus
    triangle1 = patches.Polygon([[0, equilibriumP], [equilibriumQ, equilibriumP], [0, demandFunc(0)]], closed=True, color="green", alpha=0.3)
    triangle2 = patches.Polygon([[0, equilibriumP], [equilibriumQ, equilibriumP], [0, 0]], closed=True, color="red", alpha=0.3)
    currentAxis = plt.gca()
    currentAxis.add_patch(triangle1)
    currentAxis.add_patch(triangle2)
    
    # Add labels and legend
    plt.xlabel("Quantity")
    plt.ylabel("Price")
    plt.xlim(0, 50)  # Fixed quantity range from 0 to 100
    plt.ylim(0, max(demandPrices))  # Set y-limits based on demand prices
    plt.legend()
    plt.grid(True)
    
    # Display plot
    plt.show()
    
    # Print equilibrium information
    print(f"The equilibrium price is {round(float(equilibriumP), 2)} and equilibrium quantity is {round(float(equilibriumQ), 2)}.")
    print(f"The consumer surplus at this equilibrium is {(demandFunc(0) - equilibriumP) * equilibriumQ * 0.5}")
    print(f"The producer surplus at this equilibrium is {(equilibriumP * equilibriumQ) * 0.5}")

# Example usage
Equilibrium(demandParam=50, supplyParam=20)
```

```python
import sympy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
p = sympy.Symbol("p")
def Equilibrium(demandParam, supplyParam, priceStart):
    demandEquation = demandParam - p
    # change the slope
    supplyEquation = p * (supplyParam/10)
    priceEnd = sympy.solve(demandEquation)[0]
    prices = []
    demandQ = []
    supplyQ = []
    for price in range(priceStart,priceEnd+1):
        prices += [price]
        demandQ += [demandEquation.subs(p,price)]
        supplyQ += [supplyEquation.subs(p,price)]
    
    equilibriumP = sympy.solve(demandEquation-supplyEquation)[0]
    equilibriumQ = demandEquation.subs(p,equilibriumP)
    
    
    
    triangle1 = patches.Polygon([[equilibriumQ,equilibriumP],[0,equilibriumP],[0,priceEnd]],closed=True,color="green")
    triangle2 = patches.Polygon([[equilibriumQ,equilibriumP],[0,equilibriumP],[0,0]],closed=True,color="red")
    currentAxis = plt.gca()
    currentAxis.add_patch(triangle1)
    currentAxis.add_patch(triangle2)
    
    plt.plot(demandQ,prices)
    plt.plot(supplyQ,prices)
    plt.legend(["Demand","Supply"])
    plt.plot(equilibriumQ,equilibriumP, 'ro')
    plt.xlabel("Supply and Demand Quantity")
    plt.ylabel("Price")
    plt.ylim(0, 15)
    plt.xlim(0, 10)
    plt.show()
    print("The equilibrium price is "+str(round(equilibriumP,2))+" and equilibrium quantity is "+str(round(equilibriumQ,2))+".")
    print("The consumer surplus at this equilibrium "+str((priceEnd-equilibriumP)*(equilibriumQ)*.5))
    print("The producer surplus at this equilibrium "+str((equilibriumP)*(equilibriumQ)*.5))


# you can change the range here
slider1 = widgets.IntSlider(min=5, max=15,step=1,value=10, description="Demand Intercept")
slider2 = widgets.IntSlider(min=1, max=20,step=1,value=10, description="Supply Slope")
slider3 = widgets.IntSlider(min=-5, max=5,step=1,value=0)

# Add additional text using widgets.Label or widgets.HTML
text_before = widgets.HTML(
    value="<h3>Equilibrium Simulation</h3><p>This interactive simulation allows you to explore the equilibrium price and quantity based on different demand and supply parameters.</p>"
)

text_after = widgets.HTML(
    value="<p>Adjust the <b>Demand Param</b> and <b>Supply Param</b> sliders to change the shape of the curves, and observe how the equilibrium changes.</p>"
)

# Display all elements together
ui = widgets.VBox([text_before, slider1, slider2, slider3, text_after])
out = widgets.interactive_output(Equilibrium, {'demandParam': slider1, 'supplyParam': slider2, 'priceStart': slider3})

display(ui, out)
```

## Effects of Taxes

```python
def eqSolve(eq1,eq2,tax):
    demandP = sympy.solve(eq1-q,p)[0]
    supplyP = sympy.solve(eq2-q,p)[0]
    demandP = demandP-cTax
    supplyP = supplyP+pTax

    demandQ = sympy.solve(demandP-p,q)[0]
    supplyQ = sympy.solve(supplyP-p,q)[0]
    
    return sympy.solve((demandP-supplyP, demandQ-supplyQ,tax-cTax-pTax), q,p,cTax,pTax)[q]
```

```python

p = sympy.Symbol("p")
q = sympy.Symbol("q")
cTax = sympy.Symbol("cTax")
pTax = sympy.Symbol("pTax")

def EquilibriumTax(demandParam,supplyParam,priceStart,priceEnd,tax):
    demandEquation = demandParam - p
    supplyEquation = p * (supplyParam/10)
    prices = []
    demand = []
    supply = []
    for price in range(priceStart,priceEnd+1):
        prices += [price]
        demand += [demandEquation.subs(p,price)]
        supply += [supplyEquation.subs(p,price)]
        
    
    
    nonTaxPrice = sympy.solve(demandEquation-supplyEquation)[0]
    nonTaxQ = demandEquation.subs(p,nonTaxPrice)

    
    equilibriumQ = eqSolve(demandEquation,supplyEquation,tax)
    equilibriumP1 = sympy.solve(demandEquation-equilibriumQ)[0]
    equilibriumP2 = sympy.solve(supplyEquation-equilibriumQ)[0]
    
    triangle1 = patches.Polygon([[nonTaxQ,nonTaxPrice],[equilibriumQ,nonTaxPrice],[equilibriumQ,equilibriumP1]],closed=True,color="green")
    triangle2 = patches.Polygon([[nonTaxQ,nonTaxPrice],[equilibriumQ,nonTaxPrice],[equilibriumQ,equilibriumP2]],closed=True)
    #triangle1 = patches.Polygon([[0, equilibriumP1], [equilibriumQ, equilibriumP], [0, demandFunc(0)]], closed=True, color="green", alpha=0.3)
    #triangle2 = patches.Polygon([[0, equilibriumP2], [equilibriumQ, equilibriumP], [0, 0]], closed=True, color="red", alpha=0.3)
    currentAxis = plt.gca()
    currentAxis.add_patch(triangle1)
    currentAxis.add_patch(triangle2)
    
    
    rect1 = patches.Rectangle((0,nonTaxPrice),equilibriumQ,equilibriumP1-nonTaxPrice,linewidth=1,facecolor="red")
    rect2 = patches.Rectangle((0,nonTaxPrice),equilibriumQ,equilibriumP2-nonTaxPrice,linewidth=1,facecolor="yellow")
    currentAxis.add_patch(rect1)
    currentAxis.add_patch(rect2)
    
    plt.plot(demand,prices)
    plt.plot(supply,prices)
    
    
    plt.legend([rect1,rect2,triangle1,triangle2], ["Consumer Tax","Producer Tax","Consumer Deadweight Loss","Producer Deadweight Loss"])
    plt.plot(equilibriumQ,equilibriumP1, 'ro')
    plt.plot(equilibriumQ,equilibriumP2, 'ro')
    plt.xlabel("Supply and Demand Quantity")
    plt.ylabel("Price")
    plt.ylim(0, 15)
    plt.xlim(0, 10)
    plt.show()
    print("Without Tax - the equilibrium price is "+str(round(nonTaxPrice,2))+" and equilibrium quantity is "+str(round(nonTaxQ,2)))
    print("With Tax - Price paid by consumers is "+str(equilibriumP1)+" Price received by suppliers is "+str(round(equilibriumP2,2))+" and equilibrium quantity is "+str(equilibriumQ)+".")
    print("Taxes raised from consumers equals "+str(round(equilibriumQ*(equilibriumP1-nonTaxPrice),2)))
    print("Taxes raised from producers equals "+str(round(equilibriumQ*(nonTaxPrice-equilibriumP2),2)))
    print("Total taxes raised equals "+str(equilibriumQ*tax))

slider1 = widgets.IntSlider(min=5, max=15,step=1,value=10)
slider2 = widgets.IntSlider(min=1, max=20,step=1,value=10)
slider3 = widgets.IntSlider(min=-5, max=5,step=1,value=0)
slider4 = widgets.IntSlider(min=5, max=20,step=1,value=10)
slider5 = widgets.IntSlider(min=0, max=8,step=1,value=4)
display(widgets.interactive(EquilibriumTax, demandParam=slider1, supplyParam=slider2, priceStart=slider3, priceEnd=slider4, tax=slider5))
```

## Shifts in Demand and Supply and Incidence of Tax
In this graph the second line with the suppply plus tax is not graphed but the new equilibrium point is.

## Question 3.1 -  Demand Shift - Intercept Slider
Shift the Demand Parameter - This slider is only shifting the intercept which leads to a parrallel shift in Demand. What happens to the amount of tax paid  as demand shifts out.  What happens to the ratio of tax paid from Consumer Surplus to Producer Surplus?

....  Type your answer in here

## Question 3.2 - Supply Shift - Slope Slider
Shift the Supply Parameter - This slider is shifting the slope which leads to a change in the steepness of the Supply curve. What happens to the amount of Consumer Surplus as supply becomes steeper?  What happens to the tax incidence - the ratio of tax coming from Consumer Surplus vs. Producer Surplus?

....  Type your answer in here

## Question 3.3 - Tax Shift - 
Shift the Tax Parameter - This slider is increasing or decreasing the amount of tax. 

Is it affecting the slope or the intercept? 

An increase in tax leads to and increase in the price paid by consumers. In this case how much does a given tax lead to an increase in price?


When shifting the tax alone - how does the incidence of tax between producers and consumers change?

....  Type your answer in here





