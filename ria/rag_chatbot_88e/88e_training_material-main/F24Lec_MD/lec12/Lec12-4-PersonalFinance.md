---
title: "Lec12-4-PersonalFinance"
type: lecture-notebook
week: 12
source_path: "/Users/ericvandusen/Documents/Data88E-ForTraining/F24Lec_NBs/lec12/Lec12-4-PersonalFinance.ipynb"
---

<table style="width: 100%;" id="nb-header">
    <tr style="background-color: transparent;"><td>
        <img src="https://data-88e.github.io/assets/images/blue_text.png" width="250px" style="margin-left: 0;" />
    </td><td>
        <p style="text-align: right; font-size: 10pt;"><strong>Economic Models</strong>, Fall 24 <br>
            Dr. Eric Van Dusen 
        <br>
</table>

```python
from datascience import Table
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
%matplotlib inline
```

```python
try: import yfinance as yf
except: 
    !pip install yfinance
    import yfinance as yf
```

## Personal Finance Demo 

### This notebook will demonstrate some basic concepts in personal finance, over time, using Python.
 - Rate of Savings
 - Impact of Fees on Returns over Time
 - Monte Carlo Simulation of Retirement Savings
 - Historical Returns of Stocks and Bonds


 The notebook will also motivate the use of widgets in Jupyter notebooks to interact with the data and see the impact of different parameters on the results.

 ### Part 1: Rate of Savings

```python
annual_income_widget = widgets.FloatText(
    value=50000,
    description='Annual Income:',
    style={'description_width': 'initial'}
)

savings_rate_widget = widgets.FloatSlider(
    value=0.2,
    min=0,
    max=1,
    step=0.01,
    description='Savings Rate (%):',
    style={'description_width': 'initial'}
)

years_widget = widgets.IntSlider(
    value=30,
    min=1,
    max=50,
    step=1,
    description='Years:',
    style={'description_width': 'initial'}
)

annual_return_widget = widgets.FloatSlider(
    value=0.05,
    min=0,
    max=0.2,
    step=0.01,
    description='Annual Return (%):',
    style={'description_width': 'initial'}
)
display(annual_income_widget, savings_rate_widget, years_widget, annual_return_widget)
```

Save these values as variables

```python
annual_income = annual_income_widget.value
savings_rate = savings_rate_widget.value
years = years_widget.value
annual_return = annual_return_widget.value
```

Set a formula for the rate of savings

```python
annual_savings = annual_income * savings_rate
```

Create an array of years

```python
years_array = np.arange(1, years + 1)
savings_growth = []
```

Calculate the savings for each year, create total savings appending each year's savings to the previous year's savings

```python
total_savings = 0
for year in years_array:
    # Add annual savings and apply growth
    total_savings += annual_savings
    total_savings *= (1 + annual_return)
    savings_growth.append(total_savings)
```

Create a Table to store the results

```python
savings_table = Table().with_columns(
    "Year", years_array,
    "Total Savings ($)", savings_growth
)

savings_table
```

```python

plt.figure(figsize=(10, 6))
plt.plot(years_array, savings_growth, marker='o')
plt.title("Savings Growth Over Time")
plt.xlabel("Years")
plt.ylabel("Total Savings ($)")
plt.grid(True)
plt.show()
```

## Part 2 - Retirement Planning & Fees on Investments

```python
initial_investment_widget = widgets.FloatText(
    value=10000,
    description='Initial Investment ($):',
    style={'description_width': 'initial'}
)

annual_return_widget_fee = widgets.FloatSlider(
    value=0.07,
    min=0,
    max=0.2,
    step=0.01,
    description='Annual Return (%):',
    style={'description_width': 'initial'}
)

years_widget_fee = widgets.IntSlider(
    value=30,
    min=1,
    max=50,
    step=1,
    description='Investment Duration (Years):',
    style={'description_width': 'initial'}
)

fee_percentage_widget = widgets.FloatSlider(
    value=0.01,
    min=0,
    max=0.05,
    step=0.005,
    description='Annual Fee (%):',
    style={'description_width': 'initial'}
)
second_fee_percentage_widget = widgets.FloatSlider(
    value=0.02,
    min=0,
    max=0.05,
    step=0.005,
    description='Annual Fee Comparison (%):',
    style={'description_width': 'initial'}
)

display(initial_investment_widget, annual_return_widget_fee, years_widget_fee, fee_percentage_widget, second_fee_percentage_widget)
```

Retrieve widget values for calculations

```python
initial_investment = initial_investment_widget.value
annual_return = annual_return_widget_fee.value
years = years_widget_fee.value
annual_fee = fee_percentage_widget.value
annual_fee_2 = second_fee_percentage_widget.value
```

Initialize arrays for growth with and without fees

Set starting values for investments

```python
years_array = np.arange(1, years + 1)
growth_without_fees = []
growth_with_fee_1 = []
growth_with_fee_2 = []

investment_without_fees = initial_investment
investment_with_fee_1 = initial_investment
investment_with_fee_2 = initial_investment
```

Build a growth path for the three options

```python
for year in years_array:
    # Growth without fees
    investment_without_fees *= (1 + annual_return)
    growth_without_fees.append(investment_without_fees)

    # Growth with fee 1
    net_return_1 = annual_return - annual_fee
    investment_with_fee_1 *= (1 + net_return_1)
    growth_with_fee_1.append(investment_with_fee_1)

    # Growth with fee 2
    net_return_2 = annual_return - annual_fee_2
    investment_with_fee_2 *= (1 + net_return_2)
    growth_with_fee_2.append(investment_with_fee_2)
```

Create a Table to store the results

```python
fees_table = Table().with_columns(
    "Year", years_array,
    "Growth Without Fees ($)", growth_without_fees,
    "Growth With Fees ($)", growth_with_fee_1,
    "Growth With Fees 2 ($)", growth_with_fee_2
)


fees_table.show()
```

```python
# Plot the growth over time
plt.figure(figsize=(10, 6))
plt.plot(years_array, growth_without_fees, label="Without Fees", marker='o')
plt.plot(years_array, growth_with_fee_1, label=f"With Fees ({annual_fee*100:.2f}%)", marker='o', linestyle='--')
plt.plot(years_array, growth_with_fee_2, label=f"With Fee 2 ({annual_fee_2*100:.2f}%)", marker='o', linestyle=':')
plt.title("Investment Growth Over Time with and without Fees")
plt.xlabel("Years")
plt.ylabel("Investment Value ($)")
plt.legend()
plt.grid(True)
plt.show()
```

```python

```

## Part 3: Allocation of Assets 

Let's simulate the impact of different asset allocations on the growth of investments over time.

In this example we will consider two asset classes: Stocks and Bonds.

We will use a simple model of the growth of investments in each asset class over time but with variable returns in each time period.

This model will be for a one time initial investment, but we will simulate the growth over multiple time periods.

Set up the widgets

```python
initial_investment_widget_alloc = widgets.FloatText(
    value=10000,
    description='Initial Investment ($):',
    style={'description_width': 'initial'}
)

years_widget_alloc = widgets.IntSlider(
    value=30,
    min=1,
    max=50,
    step=1,
    description='Investment Duration (Years):',
    style={'description_width': 'initial'}
)

stock_allocation_widget = widgets.FloatSlider(
    value=0.6,
    min=0,
    max=1,
    step=0.05,
    description='Stock Allocation (%):',
    style={'description_width': 'initial'}
)


display(initial_investment_widget_alloc, years_widget_alloc, stock_allocation_widget)
```

Set up average returns and standard deviations for stocks and bonds

( these are hard coded for now - but could be widgets as well)

```python
average_stock_return = 0.08 # 8% average annual return for stocks
average_bond_return = 0.03 # 3% average annual return for bonds
std_dev_stock = 0.15  # 15% annual standard deviation for stocks
std_dev_bond = 0.05   # 5% annual standard deviation for bonds
```

```python
initial_investment = initial_investment_widget_alloc.value
years = years_widget_alloc.value
stock_allocation = stock_allocation_widget.value
bond_allocation = 1 - stock_allocation
```

```python
years_array = np.arange(1, years + 1)
growth_allocation = []
```

Calculate growth based on random returns derived from asset allocation

This will use the command 'np.random.normal' to generate random returns based on the average and standard deviation of returns for stocks and bonds

The arguments to 'np.random.normal' are the average return, the standard deviation of returns, that we hard coded above.

Think about it as each year the return is a random number drawn from a normal distribution with the average return and standard deviation we set above.

This will create a random path of returns for each year, as we rerun the cell, we will get a different path of returns.  We could set a seed to get the same path of returns each time.

```python
#set_seed(42)
investment_value = initial_investment
for year in years_array:
    stock_return = np.random.normal(average_stock_return, std_dev_stock)
    bond_return = np.random.normal(average_bond_return, std_dev_bond)
    
    # Calculate the portfolio's annual return based on the allocation
    portfolio_return = (stock_allocation * stock_return) + (bond_allocation * bond_return)
    
    # Update the investment value based on the portfolio return
    investment_value *= (1 + portfolio_return)
    growth_allocation.append(investment_value)
```

Create a Table to store the results

```python
allocation_table = Table().with_columns(
    "Year", years_array,
    f"Growth with {stock_allocation*100:.0f}% Stocks / {bond_allocation*100:.0f}% Bonds ($)", growth_allocation
)

# Display the table
allocation_table.show()
```

```python

plt.figure(figsize=(10, 6))
plt.plot(years_array, growth_allocation, label=f"{stock_allocation*100:.0f}% Stocks / {bond_allocation*100:.0f}% Bonds", marker='o')
plt.title("Investment Growth Over Time with Asset Allocation (Including Volatility)")
plt.xlabel("Years")
plt.ylabel("Investment Value ($)")
plt.legend()
plt.grid(True)
plt.show()
```

## Part 4 Monte Carlo Simulation
What is the role of uncertainty in financial planning?

In the previous example, we have a single path for the growth of investments. However, in reality, the growth of investments is more complicated! 

We can use Monte Carlo simulation to simulate the growth of investments over time. This would allow us to see the range of possible outcomes for our investments.  Think of it as generating many paths of returns and seeing the range of outcomes.  This is a powerful tool for financial planning.

Step 1 - Set up the widgets

We can add a widget for the number of simulations we want to run.  This will allow us to see the range of outcomes for our investments.

```python
initial_investment_widget_alloc = widgets.FloatText(
    value=10000,
    description='Initial Investment ($):',
    style={'description_width': 'initial'}
)

years_widget_alloc = widgets.IntSlider(
    value=30,
    min=1,
    max=50,
    step=1,
    description='Investment Duration (Years):',
    style={'description_width': 'initial'}
)

stock_allocation_widget = widgets.FloatSlider(
    value=0.6,
    min=0,
    max=1,
    step=0.05,
    description='Stock Allocation (%):',
    style={'description_width': 'initial'}
)

simulations_widget = widgets.IntSlider(
    value=1000,
    min=100,
    max=5000,
    step=100,
    description='Number of Simulations:',
    style={'description_width': 'initial'}
)

# Display the widgets
display(initial_investment_widget_alloc, years_widget_alloc, stock_allocation_widget, simulations_widget)
```

Again we will hard code the average returns and standard deviations for stocks and bonds. ( for a future project we could make these widgets as well! )

```python
average_stock_return = 0.08
average_bond_return = 0.03
std_dev_stock = 0.15  # 15% annual standard deviation for stocks
std_dev_bond = 0.05   # 5% annual standard deviation for bonds
```

```python
# Retrieve widget values for calculations
initial_investment = initial_investment_widget_alloc.value
years = years_widget_alloc.value
stock_allocation = stock_allocation_widget.value
bond_allocation = 1 - stock_allocation
num_simulations = simulations_widget.value

# Initialize a list to store the final investment values of each simulation
final_values = []
```

Build a loop to simulate the growth of investments over time.  Running the simulaton multiple times is called a *Monte Carlo Simulation*.

This will be similar to the previous example, but we will run the simulation multiple times to see the range of outcomes.  The inner loop is the same as the previous example, but in the outer loop we will run the simulation multiple times.

Create a Table to store the results `final-values`

Calculate the average and standard deviation of the results `final-values`

This entire function is wrapped in a plot function that will plot the results of each run of the Monte Carlo simulation.  Think of it as 1000 different plots that map out the range of outcomes for the investments.  

In the plot we have set the alpha to 0.1 so that we can see the overlap of the different paths.  Alpha is a parameter that sets the transparency of the plot.  So, the lower the alpha, the more transparent the plot.  


We will also plot the average and standard deviation of the results.  This will give us a sense of the range of outcomes for our investments.

```python

plt.figure(figsize=(12, 8))
#  Outer loop for running multiple simulations
for _ in range(num_simulations):
    investment_value = initial_investment
    growth_path = []

    # Inner loop for each simulation
    for year in range(years):
        # Generate random returns for stocks and bonds based on mean and standard deviation
        stock_return = np.random.normal(average_stock_return, std_dev_stock)
        bond_return = np.random.normal(average_bond_return, std_dev_bond)
        
        portfolio_return = (stock_allocation * stock_return) + (bond_allocation * bond_return)
      
        investment_value *= (1 + portfolio_return)
        growth_path.append(investment_value)

    # Store the final investment value for this simulation
    final_values.append(investment_value)

    # Plot this simulation's growth path
    plt.plot(range(1, years + 1), growth_path, color='blue', alpha=0.05) # alpha is the transparency level

# Calculate summary statistics of the final values across all simulations
mean_final_value = np.mean(final_values)
median_final_value = np.median(final_values)
percentile_10 = np.percentile(final_values, 10)
percentile_90 = np.percentile(final_values, 90)

# Plot the summary statistics on the chart
plt.title("Monte Carlo Simulation of Investment Growth with Asset Allocation")
plt.xlabel("Years")
plt.ylabel("Investment Value ($)")
plt.grid(True)
plt.axhline(mean_final_value, color='green', linestyle='--', label=f"Mean Final Value: ${mean_final_value:,.2f}")
plt.axhline(median_final_value, color='orange', linestyle='--', label=f"Median Final Value: ${median_final_value:,.2f}")
plt.axhline(percentile_10, color='red', linestyle='--', label=f"10th Percentile: ${percentile_10:,.2f}")
plt.axhline(percentile_90, color='purple', linestyle='--', label=f"90th Percentile: ${percentile_90:,.2f}")
plt.legend()
plt.show()
```

## Appendix (Part 4)  - get the parameters for return and sd from the markets

We can get the average returns and standard deviations for stocks and bonds from the markets.  This will allow us to use real data in our simulations.  

We have hard coded the values for now, but we could use the market values to update our hard coded values above   This would allow us to use real data in our simulations.

We will use the `yfinance` package to get the data from the markets.  This package allows us to get data from Yahoo Finance in a simple way, no API key, and in the format we want.  

We will use the `yfinance` package to get the data for the **S&P 500** and the **10 year treasury** bond.  We will call the tickers for these assets `^GSPC` and `^TNX` respectively. 

We will use the data to calculate the average annual returns and standard deviations for stocks and bonds. Using annual returns and standard deviations is clearly one limitation of this model, but it is a simple way to get a sense of the range of outcomes for our investments.

There are two important choices in the following code:
- The time period for the data
- The index for the data

*Note for Data 88E: There is a little Pandas manipulaton in the code to calculate the average annual returns and standard deviations for stocks and bonds.  We will use the `ffill` and `resample` methods to fill in any missing values and to resample the data to get the annual returns and standard deviations.  We will use the `dropna` method to remove any missing values from the data.*

```python
start_date = "2000-01-01"
end_date = "2023-01-01"
```

```python
# Fetch historical data for S&P 500 (stocks) and TLT (bonds) using yfinance
stock_data = yf.download('^GSPC', start=start_date, end=end_date)
bond_data = yf.download('TLT', start=start_date, end=end_date)

# Convert to annual adjusted close prices by resampling to year-end and forward-filling
stock_data = stock_data['Adj Close'].resample('Y').ffill()
bond_data = bond_data['Adj Close'].resample('Y').ffill()

# Calculate annual returns as percentage change for stocks and bonds
stock_returns = stock_data.pct_change().dropna()
bond_returns = bond_data.pct_change().dropna()
```

```python
# Create datascience Tables for annual returns
stock_table = Table().with_columns("Year", stock_data.index.year[1:], "Stock Returns", stock_returns.values)
bond_table = Table().with_columns("Year", bond_data.index.year[1:], "Bond Returns", bond_returns.values)
```

```python
# Calculate mean and standard deviation using the Table methods
average_stock_return_data = stock_table.column("Stock Returns").mean()
std_dev_stock_data = stock_table.column("Stock Returns").std()
average_bond_return_data = bond_table.column("Bond Returns").mean()
std_dev_bond_data = bond_table.column("Bond Returns").std()
```

```python
# Display the results
print(f"Average Stock Return (Market Data): {average_stock_return_data * 100:.2f}%")
print(f"Stock Return Standard Deviation (Market Data): {std_dev_stock_data * 100:.2f}%")
print(f"Average Bond Return (Market Data): {average_bond_return_data * 100:.2f}%")
print(f"Bond Return Standard Deviation (Market Data): {std_dev_bond_data * 100:.2f}%")
```

## Updating the Monte Carlo Simulation with Market Data !?

- How do these results compare to the parameters that we hard coded above?
- What are the implications for financial planning?
- How do the results change if we use real data from the markets?

```python
print( " Actual - Data")
print(f"Difference in Average Stock Return: {(average_stock_return - average_stock_return_data) * 100:.2f}%")
print(f"Difference in Stock Return Standard Deviation: {(std_dev_stock - std_dev_stock_data) * 100:.2f}%")
print(f"Difference in Average Bond Return: {(average_bond_return - average_bond_return_data) * 100:.2f}%")
print(f"Difference in Bond Return Standard Deviation: {(std_dev_bond - std_dev_bond_data) * 100:.2f}%")
```

Looks like our hard-coded values were a bit optimistic?  
- We overestimanted returns
- We underestimated volatility

```python

```

```python

```

```python

```

