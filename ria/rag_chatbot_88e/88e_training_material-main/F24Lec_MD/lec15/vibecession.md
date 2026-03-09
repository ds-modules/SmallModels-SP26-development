---
title: "vibecession"
type: lecture-notebook
week: 15
source_path: "/Users/ericvandusen/Documents/Data88E-ForTraining/F24Lec_NBs/lec15/vibecession.ipynb"
---

##  Vibecession - The Notebook!
The purpose of this notebook is to get the data and check the vibes behind the post-COVID drop in Consumer Sentiment

The concept of vibecession and the testing it in a notebook folllows from a couple key sources
 - A Tweet by **Quantian** that [showed a test of the hypothesis](https://threadreaderapp.com/thread/1688397994821873664.html#google_vignette) - that this notebook is recreating

- Which was followed up by a FT recreation - for multiple countries  - and how partisan this gap is !  You can read about it here [Should we believe Americans when they say the economy is bad? John Burn-Murdoch](https://www.ft.com/content/9c7931aa-4973-475e-9841-d7ebd54b0f47)

### But our starting point can be a more accessible commentator 


[Kyla Scanlon](https://kylascanlon.com/), an online content creator and independent economics educator, coined the term "vibecession" to describe a phenomenon where public sentiment about the economy is overwhelmingly negative, despite relatively positive economic indicators like GDP growth and low unemployment rates. The term is a portmanteau of "vibes" and "recession," suggesting an economic downturn driven primarily by negative public sentiment rather than direct economic metrics.

The concept of the vibecession was introduced by Scanlon in a June 2022 newsletter, amidst observations that, while hard economic data was showing signs of stability and growth, the general sentiment among the public remained pessimistic. This disconnect, where the public feels economically insecure despite positive indicators, has been a central theme in Scanlon's discussions on platforms like TikTok, where she aims to make complex economic concepts more accessible and engaging to a broader audience, particularly younger people.


[Kyla Scanlon](https://www.marketplace.org/2022/09/07/for-tiktok-maker-kyla-scanlon-its-about-making-finance-fun-and-a-bit-chaotic/) on Marketplace

```python
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
import plotly.express as px
```

```python
# try to import fredapi
try:
    from fredapi import Fred
except ImportError:
    !pip install fredapi
    from fredapi import Fred

# try to import yfinance
try:
    import yfinance as yf
except ImportError:
    !pip install yfinance
    import yfinance as yf
fred = Fred(api_key='e3053cdc3e94dfb2b73c5945b0d1b1f7')
```

##  Part  1 - Gather the Data 
 - The following sections are how we source the data
  - This may be too boring and pedantic  - but they are related to the learnign outcomes of the class 
  - We will use a combination of the python package `fredapi` to dowload a few time series from the Federal Reserve Bank of St Louis
  - And python package `yfinance` (derived from Yahoo Finance)  to get additional series
  - An enterprising student could add additional series to the modeling process
*Skip to Part 2 if you just want to see the results*

## Consumer Sentiment 

The [University of Michigan Consumer Sentiment Index](http://www.sca.isr.umich.edu/) is an economic indicator that assesses the confidence, conditions, and expectations of U.S. consumers regarding their financial situation and the general state of the economy. The index is based on a monthly survey of approximately 500 households regarding their personal finances, business conditions, and buying conditions. It is divided into two parts: the Index of Consumer Expectations and the Current Economic Conditions Index.

This data is significant as it can provide insights into consumer behavior, which helps in predicting changes in spending and saving habits. Higher consumer confidence typically indicates that people feel secure in their personal financial situation and thus are more likely to increase spending, which is a key driver of economic growth.

The index is often used by analysts and policymakers to understand consumer sentiment and its potential impact on the economy. For instance, rising sentiment can suggest increased consumer spending and economic expansion, while declining sentiment might indicate economic slowdowns or recessions.

```python
# This is the data we want to model, we are going to get the UMCSENT series from FRED
UMCSENT = fred.get_series('UMCSENT', observation_start='1979-01-01', observation_end='2024-03-01')
UMCSENT
print(UMCSENT.index.tzinfo)
UMCSENT
```

Lets take a look at the data that we are trying to model 

The time range is roughly the last 40 years

```python
fig = px.line(UMCSENT, title='Consumer Sentiment Index')
fig.update_yaxes(range=[0, 120])
fig.show()
```

###  Get some Features eg Explanatory Variables for our Model

Now we want to search for some basic economics variables that could explain how Consumers are Feeling 

Variables that were suggested by Quantian on Twitter were the following 

- Inflation rate
- Inflation rate change
- Unemployment
- Unemployment change
- Housing prices
- Real wages
- Dollar strength
- Interest rates
- Stock prices

For the first set of Series we will go FRED and download them according to the series name

```python
#Unemployment rate
UNRATE=fred.get_series('UNRATE', observation_start='1979-01-01', observation_end='2024-03-01')
#Inflation rate
CPIAUSCL= fred.get_series('CPIAUCSL', observation_start='1979-01-01', observation_end='2024-03-01')
#GDP
GDP=fred.get_series('GDP', observation_start='1979-01-01', observation_end='2024-03-01')
#housing price change
USSTHPI=fred.get_series('USSTHPI', observation_start='1979-01-01', observation_end='2024-03-01')
#intereste rate
FEDFUNDSS=fred.get_series('FEDFUNDS', observation_start='1979-01-01', observation_end='2024-03-01')
#Real Wages
WAGES=fred.get_series('LES1252881600Q', observation_start='1979-01-01', observation_end='2024-03-01')

# The following Series ARE available on FRED but not with the time range we need
#S&P 500
#SP500=fred.get_series('SP500', observation_start='1979-01-01', observation_end='2024-03-01')
#Dolar index monthly
#DTWEXM=fred.get_series('DTWEXM', observation_start='1979-01-01', observation_end='2024-03-01')
# inflation change quarter to quarter
#INFCHANGE=fred.get_series('BPCCRO1Q156NBEA', observation_start='2019-01-01', observation_end='2024-03-01')
```

### Processing the data 

- In the next few cells I will take a look at each series we have downloaded 
- Some have to check the Time Zone - I am going to go with making them all "time zone naive"
- Some data are monthly, some are quarterly - I am going to adjust everything to monthly by filling in quarterly data for the first of each month in that quarter

```python
#Inflation rate
print(CPIAUSCL.index.tzinfo)
CPIAUSCL
```

```python
#housing price change index is quarterly, we need to resample it to monthly
USSTHPI=USSTHPI.resample('MS').ffill()
print(USSTHPI.index.tzinfo)

USSTHPI
```

```python
# remove this series 
#DTWEXM=DTWEXM.resample('MS').ffill()
#print(DTWEXM.index.tzinfo)

#DTWEXM
```

```python
# Wage series is quarterly, we need to resample it to monthly
WAGES=WAGES.resample('MS').ffill()
print(WAGES.index.tzinfo)

WAGES
```

```python
#convert gdp to monthly by using the quarterly data
GDP=GDP.resample('MS').ffill()
print(GDP.index.tzinfo)

GDP
```

```python
# Make a new variable thats the percent change in CPI
cpichange = CPIAUSCL.pct_change()
print(cpichange.index.tzinfo)

cpichange
```

### For the next couple of series we can get a longer time series by going to YFinance



```python
#download S&P 500 data close price only
SP500 = yf.download('^GSPC', start='1979-01-01', end='2024-03-01')
#convert daily data to monthly data
SP500=SP500.resample('MS').mean()
#drop all columns except close price
SP500=SP500['Close']
#make time zone naive
SP500.index = SP500.index.tz_localize(None)

SP500
```

```python
# Get the Dollar Index data
DXY = yf.download('DX-Y.NYB', start='1979-01-01', end='2024-03-01')
DXY=DXY.resample('MS').mean()
DXY=DXY['Close']
DXY.index = DXY.index.tz_localize(None)
DXY
```

Now Lets Combine these series into a Dataframe called Vibes
- save it as a csv so we can skip all the data processing in the future
- Drop missing values for now (What gets dropped!?!) ( eg data before Nov 2023)
- check the data visually

```python
# create a dataframe using pd.concat
vibes = pd.concat([UMCSENT, UNRATE, CPIAUSCL, GDP, USSTHPI, FEDFUNDSS, SP500, DXY, cpichange,WAGES], axis=1) 
vibes.columns = ['UMCSENT', 'UNEMPLOYMENT', 'CPI', 'GDP', 'HOUSINGPRICE', 'FEDFUNDS', 'SP500', 'DOLLAR', 'CPICHANGE','WAGES'] 
# drop any rows with missing values
vibes.dropna(inplace=True)
# save vibes to a csv file
vibes.to_csv('vibes.csv')
vibes
```

## Part 2  - Modeling Consumer Sentiment with Macroeconomic Data Series 

In the next section we will be modeling consumer sentiment and using the macroeconomcis time series 
- split up the data into before and after covid
- run a regression to predict consumer sentiment
- Compare predicted to actual outcomes
- Look at the residuals
- Show how to run the model in SKLearn instead of statsmodels

```python
#read in vibes.csv
vibes = pd.read_csv('vibes.csv', index_col=0)
# Y = the variable we want to predict, the Target in ML
# X = the explanatory variabbles we use to predicy Y , or features in ML
X = vibes[['UNEMPLOYMENT', 'CPI', 'GDP', 'HOUSINGPRICE', 'FEDFUNDS', 'SP500', 'DOLLAR', 'CPICHANGE','WAGES']]
Y = vibes['UMCSENT']  # Make sure this is the correct column name for Consumer Sentiment
```

The first model will use Statsmodels to run a simple linear regression 

*I know there are problems with this model specification!*

```python
X = sm.add_constant(X)
# Fit the model
model = sm.OLS(Y, X).fit()
# Print out the statistics
print(model.summary())
```

### Redo the Model as before and after COVID

The idea here is to train the model toon data up until 2019 and then use it to predict 2020-2024

```python
# Split the data into training and testing sets before and after Dec 2019
vibes_train = vibes.loc[:'2019-12-01']
vibes_test = vibes.loc['2020-01-01':]

X_train = vibes_train[['UNEMPLOYMENT', 'CPI', 'GDP', 'HOUSINGPRICE', 'FEDFUNDS', 'SP500', 'DOLLAR', 'CPICHANGE','WAGES']]
Y_train = vibes_train['UMCSENT']

X_test = vibes_test[['UNEMPLOYMENT', 'CPI', 'GDP', 'HOUSINGPRICE', 'FEDFUNDS', 'SP500', 'DOLLAR', 'CPICHANGE','WAGES']]
Y_test = vibes_test['UMCSENT']

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)
```

```python
# Fit the model on the training data
model = sm.OLS(Y_train, X_train).fit()
# Summary of the model
print(model.summary())
# Predict the test data
Y_pred = model.predict(X_test)
# Calculate the MAE
mae = mean_absolute_error(Y_test, Y_pred)
print(f'The Mean Absolute Error of the model is {mae}')
```

```python
#plot the actual vs predicted values over the testing data
plt.figure(figsize=(12, 6))
plt.plot(Y_train, label='Train')
plt.plot(Y_test, label='Test')
plt.plot(Y_pred, label='Predicted')
plt.ylabel('Consumer Sentiment Index')
plt.legend()
plt.show()
```

```python
# Plot the data and the model's prediction for the entire time period
#X = vibes[['UNEMPLOYMENT', 'CPI', 'GDP', 'HOUSINGPRICE', 'FEDFUNDS', 'SP500', 'DOLLAR', 'CPICHANGE','WAGES']]
#X = sm.add_constant(X)
#Y = vibes['UMCSENT']
Y_pred_all = model.predict(X)
plt.figure(figsize=(12, 6))
plt.plot(Y, label='Actual')
plt.plot(Y_pred_all, label='Predicted')
# add a line at March 2020 
plt.axvline('2020-03-01', color='red', linestyle='--')
#Label the Y axis
plt.ylabel('Consumer Sentiment Index')
# label the x axis by every 5 years
plt.xticks(['1980-01-01', '1985-01-01', '1990-01-01', '1995-01-01', '2000-01-01', '2005-01-01', '2010-01-01', '2015-01-01', '2020-01-01'])

plt.legend()
plt.show()
```

```python
# Plot the data and the model's prediction for the entire time period using plotly
# Plot actual and predicted values for Y
fig = px.line(vibes, y=['UMCSENT'], title='Consumer Sentiment Index')
fig.add_scatter(x=vibes.index, y=Y_pred_all, mode='lines', name='Predicted')
fig.update_yaxes(range=[0, 120])
# add a line at March 2020
fig.add_vline(x='2020-03-01', line_dash='dash', line_color='red')
# add y axis label
fig.update_yaxes(title_text='Consumer Sentiment Index')
fig.show()
```

```python
#plot the actual vs predicted values
plt.plot(Y_test.index, Y_test, label='Actual')
plt.plot(Y_test.index, Y_pred, label='Predicted')
plt.legend()
plt.show()
```

```python
#plot the residuals
residuals = Y_test - Y_pred
plt.plot(Y_test.index, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.show()
```

```python
#plot the residuals over the entire time period 

residuals_all = Y - Y_pred_all
plt.plot(Y.index, residuals_all)
plt.axhline(0, color='red', linestyle='--')
# label the x axis by every 5 years
plt.xticks(['1980-01-01', '1985-01-01', '1990-01-01', '1995-01-01', '2000-01-01', '2005-01-01', '2010-01-01', '2015-01-01', '2020-01-01'])

plt.show()
```

```python
# Lets make a model using sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
```

```python
# SKlearn Linear Regression Model
# Create a linear regression model
lr = LinearRegression()

# Fit the model
lr.fit(X_train, Y_train)

# Predict on the test set
Y_pred = lr.predict(X_test)

# Calculate the MAE
mae = mean_absolute_error(Y_test, Y_pred)
print(f'The Mean Absolute Error of the model is {mae}')

# print the R^2 value
print(f'The R^2 value of the training model is {lr.score(X_train, Y_train)}')
print(f'The R^2 value of the model on test data is {lr.score(X_test, Y_test)}')
```

```python
#plot the actual vs predicted values
plt.plot(Y_test.index, Y_test, label='Actual')
plt.plot(Y_test.index, Y_pred, label='Predicted')
plt.legend()
plt.show()
```

```python
# Fit the model
model = LinearRegression()
model.fit(X_train, Y_train)
# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
mae = mean_absolute_error(Y_test, y_pred)
# Print the MAE
print(f'The Mean Absolute Error of the model is {mae}')
# print the R^2
print(f'The R^2 of the model on test data is {model.score(X_test, Y_test)}')
```

```python
#Plot y  over time with years on the x-axis
plt.figure(figsize=(12,6))
plt.plot(Y.index, Y, label='Actual')
plt.plot(Y_test.index, y_pred, label='Predicted')
plt.ylabel('Consumer Sentiment Index')
plt.legend()
plt.show()
```

##  SKlearn - Let's try a different ML model
Now that we have the ML model all set up, we can try a different ML model.  One that has been suggested for time series for improved fits is the Random Forest Regressor

```python
from sklearn.ensemble import RandomForestRegressor

# Create the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
model.fit(X_train, Y_train)

# Predict on the test set
RFRpred = model.predict(X_test)

#Evaluate the model on traning data
r2 = model.score(X_train, Y_train)
print("R-squared on training data :", r2)

# Calculate the MAE
mae = mean_absolute_error(Y_test, RFRpred)
print(f'The Mean Absolute Error of the model is {mae}')

# Evaluate the model (using R^2 score here)
r2 = model.score(X_test, Y_test)
print("R-squared on test data :", r2)
```

Let's look a the plots as we did for the regressions above

```python
#Plot y  over time with years on the x-axis
plt.figure(figsize=(12,6))
plt.plot(Y.index, Y, label='Actual')
plt.plot(Y_test.index, RFRpred, label='Predicted')
plt.ylabel('Consumer Sentiment Index')
plt.legend()
plt.show()
```

```python
#Plot the Y_test and the predicted values
plt.plot(Y_test.index, Y_test, label='Actual')
plt.plot(Y_test.index, RFRpred, label='Predicted')
plt.legend()
plt.show()
```

```python
# plot the Y and the predicted values over the entire time period
RFRpred_all = model.predict(X)
plt.figure(figsize=(12, 6))
plt.plot(Y, label='Actual')
plt.plot(RFRpred_all, label='Predicted')
plt.ylabel('Consumer Sentiment Index')
plt.legend()
plt.show()
```

```python
# Plot the feature importances
importances = model.feature_importances_
plt.bar(X.columns, importances)
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.show()
```



```python

```

