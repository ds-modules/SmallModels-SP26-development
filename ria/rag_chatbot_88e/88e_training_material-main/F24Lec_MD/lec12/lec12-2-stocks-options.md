---
title: "lec12-2-stocks-options"
type: lecture-notebook
week: 12
source_path: "/Users/ericvandusen/Documents/Data88E-ForTraining/F24Lec_NBs/lec12/lec12-2-stocks-options.ipynb"
---

<table style="width: 100%;">
    <tr style="background-color: transparent;"><td>
        <img src="https://data-88e.github.io/assets/images/blue_text.png" width="250px" style="margin-left: 0;" />
    </td><td>
        <p style="text-align: right; font-size: 10pt;"><strong>Economic Models</strong>, Fall 2024<br>
            Dr. Eric Van Dusen <br>
        Sreeja Apparaju <br>
        Kidong Kim</p></td></tr>
</table>

# Lecture 12: Finance

## This notebook takes a look at some simple tools for looking at the stock market
 - Previously Yahooo finance had a free API for reading in historical data on stocks
 - However the Yahoo API got discontiued
 - An awesome quant made a python package that recreated this functionality by scraping the information
 
Check out the documentation for [Yfinance package](https://pypi.org/project/yfinance/)

The package - called yfinance is not on the datahub so first we need to install it

```python
try:
    import yfinance as yf
except:
    !pip install yfinance
    import yfinance as yf
```

```python
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from datetime import timedelta, date, datetime
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import display
import warnings
from datascience import *
warnings.filterwarnings('ignore')
%matplotlib inline
```

## S&P 500 and the Nasdaq

The yfinance package allows us to download by stock ticker and make a Pandas Dataframe - here we will pull in by the market-wide tickers for the S&P 500 and the Nasdaq

```python
data_SPNQ = yf.download(("^GSPC", '^IXIC'), start="1993-01-29", end="2022-04-05")
```

The following section uses the dataframe to build out a new dataframe with returns - the amount earned each day on the previous days close

```python
data_SN = data_SPNQ.iloc[:, [2,3]]
data_SP =data_SPNQ.iloc[:, 0]
data_NQ = data_SPNQ.iloc[:, 1]
dSP = np.array(len(data_SP)-1)
for i in range(len(data_SP)-1):
    dat = ((data_SP[i] - data_SP[i+1])/data_SP[i])*100
    dSP = np.append(dSP,dat)
dNQ = np.array(len(data_NQ)-1)
for i in range(len(data_NQ)-1):
    dat = ((data_NQ[i] - data_NQ[i+1])/data_NQ[i])*100
    dNQ = np.append(dNQ,dat)
data_SN['SP Returns'] = dSP
data_SN['NQ Returns'] = dNQ
```

```python
data_SN.iloc[:,[0,1]].plot(color = ('blue', 'red'), figsize=(10,8), alpha =0.3);
```

```python
data_SN[['SP Returns', 'NQ Returns']].iloc[1:].plot(color = ('blue', 'red'), figsize=(10,8), alpha = 0.3);
```

```python
data_SN.iloc[:,[0,1]].plot(color = ('blue', 'red'), figsize=(10,8), alpha =0.3);
```

```python
data_SN[['SP Returns', 'NQ Returns']].iloc[1:].plot(color = ('blue', 'red'), figsize=(10,8), alpha = 0.3);
```

## Let's dive deeper into the Yfinance API and and work with the data

First we will define three stocks that we want to look at more closely, and examine what sort of information we can get for each stock.  

Lets look at 
 - Twitter
 - Tesla
 - USO - an ETF (exchange traded fund) that tracks the price of oil

```python
twitter_ticker = yf.Ticker("twtr")
tesla_ticker = yf.Ticker("tsla")
uso_ticker = yf.Ticker("uso")
```

There is actually a lot of information that yfinance API can provide for any equity.  In the example above we only downloaded the closing price for each of the indexes.

```python
#tesla_ticker.info
```

## Out of all this info - let's extract the stock prices

This will put the dates, prices, and volumes into a *Pandas* dataframe with the name of the stock

```python
#twitter = twitter_ticker.history(period="max")
tesla = tesla_ticker.history(period="max")
uso = uso_ticker.history(period="max")
```

```python
tesla
```

## Lets look at the market for Options for Twitter 
 - This will show us the possible strike dates for different options
 - From short term - this week - to long term - in two years

```python
tesla_ticker.options
```

## Downloading Calls and Puts 
Let's download all of the Calls and Puts for Tesla  into two tables

```python
tesla_options = tesla_ticker.option_chain(tesla_ticker.options[0])
```

```python
tesla_options.calls
```

```python
# Returns two table 0 : calls, 1 : puts
tesla_options = tesla_ticker.option_chain(tesla_ticker.options[0])
tesla_put , tesla_call = tesla_options.puts, tesla_options.calls
relevant_columns = ['lastPrice', 'change', 'percentChange', 'volume', 'strike']
tesla_put, tesla_call = tesla_put[relevant_columns], tesla_call[relevant_columns]
tesla_put.describe(), tesla_call.describe()
```

## Let's put these together into a joint table that are joined by the strike price

```python
tesla_option = pd.merge(tesla_put, tesla_call, how='inner', on = "strike")
#tesla_option = pd.merge(tesla_put, tesla_call, how='outer', on = "strike")
tesla_option[12:32]
```

## Now lets code up a graph of the puts and calls for Tesla

```python
current_tesla_p = tesla.iloc[-1]['Close']
current_tesla_p

plt.figure().set_size_inches(15, 5)

plt.title("Tesla calls vs puts",  fontsize=15)

plt.plot(tesla_option['strike'], tesla_option['lastPrice_x'], color='r', label='call', linewidth=3)
plt.plot(tesla_option['strike'], tesla_option['lastPrice_y'], color='b', label='put', linewidth=3)
plt.axvline(x = current_tesla_p, color = 'g', label = 'Current Price', linewidth=4) #Current Price

plt.axis([tesla_option['strike'].iloc[0], tesla_option['strike'].iloc[-1], 0, max(max(tesla_option['lastPrice_x']), max(tesla_option['lastPrice_y']))])
plt.legend()
```

```python
def option(ticker):
  options_obj = ticker.option_chain(ticker.options[0])
  put, call = options_obj.puts, options_obj.calls
  relevant_columns = ['lastPrice', 'change', 'percentChange', 'volume', 'strike']
  put, call = put[relevant_columns], call[relevant_columns]
  option_sheet = pd.merge(put, call, how='inner', on = "strike")
  return option_sheet

option(uso_ticker)
```

## QuantStats Package
The same developer made a more recent package that draws on Yfinance but makes a whole set of summary tables 

Check out the documentation for the [QuantStats Package](https://pypi.org/project/QuantStats/)

```python
try:
    import quantstats as qs
except:
    !pip install quantstats
    import quantstats as qs
```

```python
import quantstats as qs

# extend pandas functionality with metrics, etc.
qs.extend_pandas()

# fetch the daily returns for a stock
stock = qs.utils.download_returns('TSLA')

# show sharpe ratio
qs.stats.sharpe(stock)

# or using extend_pandas() :)
stock.sharpe()
```

### QuantStats can make a "Snapshot" of stock performance

```python
qs.plots.snapshot(stock, title='Tesla Performance')
```

## Relevant materials and sources

https://algotrading101.com/learn/yfinance-guide/ <br>
https://pypi.org/ <br>
https://pypi.org/project/QuantStats/

