---
title: "macro-fred-api"
type: lecture-notebook
week: 8
source_path: "/Users/ericvandusen/Documents/Data88E-ForTraining/F24Lec_NBs/lec08/macro-fred-api.ipynb"
---

<table style="width: 100%;" id="nb-header">
    <tr style="background-color: transparent;"><td>
        <img src="https://data-88e.github.io/assets/images/blue_text.png" width="250px" style="margin-left: 0;" />
    </td><td>
        <p style="text-align: right; font-size: 10pt;"><strong>Economic Models</strong>, EdX<br>
            Dr. Eric Van Dusen <br>
            Alan Liang <br> 
            Umar Maniku <br> 
            Matthew Yep <br> 
            Yiyang Chen <br> 
            Bennett Somerville <br>
        Akhil Venkatesh <br>
</table>

# Lecture Notebook 4.3: Macroeconomic Indicators

```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

```python
# Load our API key from the .env file
def load_environment_variables(filename):
    with open(filename) as f:
        for line in f:
            (key, value) = line.replace('\n', '').split('=')
            os.environ[key] = value

load_environment_variables('.env')
```

**Learning Objectives:**  
In this notebook, we will download macroeconomic data and examine how the models and theories we discussed in lecture fit with these real life data. This notebook will introduce the basic concept of an API, the way to use it for economic research and study.

## Macroeconomic Data and Fred API

We will start by first download some macroeconomic data from FRED. In this notebook, we will introduce two ways of interacting with online sources of data with API (Application Programming Interface).

**What is an API?**  
In contrast to a user interface, which connects a computer to a person, an application programming interface (API) connects computers or pieces of software to each other. It is not intended to be used directly by a person (the end user) other than a computer programmer who is incorporating it into the software. An API is often made up of different parts which act as tools or services that are available to the programmer. A program or a programmer that uses one of these parts is said to call that portion of the API. The calls that make up the API are also known as subroutines, methods, requests, or endpoints. An API specification defines these calls, meaning that it explains how to use or implement them. 

In basic terms, APIs just allow applications to communicate with one another. For the APIs we are concerned right now--web based APIs that return data in response to a request made by us--**they allow us to get data from outside sources by sending an API a request detailing the information we want. Then the API will "respond" with the requested data to us.**

![What is an API](what_is_an_API.png)
Source: Perry Eising, "What exactly is an API?"

### General Way of Interacting with an API

First, we will use the most common way to access the data through an API without using any packages other may have already built for a particular site. Usually this is the way you would interact with an API.

#### Step 1: Get the API Key
**In most cases, you will need to get an API key in order to access an API.** For many resources, it involves some paperwork to apply and/or limited free usage, so it is good practice to keep your API keys private as long as it is possible. In this lecture notebook, we will use macroeconomic data from FRED, which is one of the most famous and convenient sources of economic data. For FRED, the process of getting an API key is simple. Request the API key [here](https://fred.stlouisfed.org/docs/api/api_key.html).

```python
# Because API keys are private, we store our API key in the .env file in this directory, which is not published on GitHub.
api_key = os.environ['API_KEY']
```

#### Step 2: Learn to use the API
Using an API is like ordering food at a restaurant with a menu. To have a delicious meal, we have to know what food the restaurant offers, and any other additional information (for example, how would you like your steak). Similarly, it is very important for us to know what requests an API take through the API documentation. **The API documentation will inform us about how we can use specific instructions to obtain the data that we want, and what the returned data would look like.** Look up the Fred API's documentation [here](https://fred.stlouisfed.org/docs/api/fred/series_observations.html).

<div class="alert alert-info">
<b> Example: "api.stlouisfed.org/fred/series/observations?series_id=GNPCA&api_key=abcdefghijklmnopqrstuvwxyz123456" <br>  
  
Endpoint: "api.stlouisfed.org/fred/series/observations"  <br>
Parameters: series_id=GNPCA, api_key=abcdefghijklmnopqrstuvwxyz123456
</div>

Try the following link, replacing REDACTED with your API key to see what the API will return!   

https://api.stlouisfed.org/fred/series/observations?series_id=GNPCA&api_key=REDACTED&file_type=json

#### Step 3: Make the fetch
Now we are ready to start writing code to fetch the data we want through the API.

```python
import requests
from urllib.parse import urlencode
endpoint = "https://api.stlouisfed.org/fred/series/observations"
```

Parameters that we would like to include in our query (these can be found in the FRED API documentations!):  
- `series_id`: The Economic data series we want (e.g. GDP)
- `observation_start`: The earliest date we want our data to include
- `observation_end`: The latest date we want our data to include
- `frequency`: The frequency of observations (e.g. Annually, Quarterly, Monthly)
- `units`: The units of observations (e.g. plain numbers, percentage change from a year ago, etc.)
- `api_key`: The API key
- `file_type`: The file data that will returned from FRED (e.g. json, xml)

```python
def fetch(series_id, unit="lin"):
    # specifies parameters
    params = {"series_id": series_id, 
              "observation_start": "1958-01-01", 
              "observation_end": "2023-10-01",
              "frequency": "q", # quarterly
              "units": unit,
              "api_key": api_key, 
              "file_type": "json"
             }
    
    # forms API request
    url_params = urlencode(params)
    url = f"{endpoint}?{url_params}"
    
    # fires off the request
    res = requests.get(url)
    
    # checks if the request encounters an error
    if res.status_code not in range(200, 299):
        raise Exception(f'Fetch request for "{series_id}" failed (Error: {res.status_code})')
    
    # return the content of the response
    return res.json()
```

Let's try downloading the real GDP data from Fred now.

```python
res_json_GDP = fetch("GDPC1")

# process the response
gdp = pd.DataFrame(res_json_GDP["observations"])[["date", "value"]]
gdp.rename(columns={"value": "GDP"}, inplace=True)
gdp['date'] = pd.to_datetime(gdp['date'], format='%Y-%m-%d')
gdp['GDP'] = pd.to_numeric(gdp['GDP'])
gdp
```

```python
plt.plot(gdp['date'], gdp['GDP'])
```

**Congratulations! We have just made our first request to Fred API and obtain some useful data.**

Now let's try download more macroeconomic indicators.

```python
# this function allows us to fetch and process the data together
def fetch_and_process(series_id, unit="lin"):
    res_json = fetch(series_id, unit)
    return pd.DataFrame(res_json["observations"])[["date", "value"]].rename(columns={"date":"DATE", "value": f"{series_id}_{unit}"})
```

```python
gdp = fetch_and_process("GDPC1")
gdp_growth = fetch_and_process("GDPC1", "pc1")
inflation = fetch_and_process("CPIAUCSL", "pc1")
unemployment = fetch_and_process("UNRATE")
ffr = fetch_and_process("DFF")
nrou = fetch_and_process("NROU")
core_inflation = fetch_and_process("CPILFESL", "pc1")
potential_gdp = fetch_and_process("GDPPOT")

gdp_growth
```

### Site-Specific Prebuilt Packages

Now we will switch gears to use some prebuilt packages to access FRED API. Notice that the prebuilt packages are not necessarily available for every API. But here we will use the "FredAPI" package developed by Mortada Mehyar. Documentation [here](https://github.com/mortada/fredapi).

```python
# may need to install the package and its dependencies
!pip install fredapi
from fredapi import Fred
fred = Fred(api_key=api_key)
GDP_fredapi = pd.DataFrame(fred.get_series('GDP'))
GDP_fredapi
```

You will get more practice with FREDAPI package in the lab!

## Macroeconomic Indicators and Data

**Now we're off to use the data we have just got!**

```python
# merge all macroeconomic data we have got!
macroeconomics = gdp.merge(gdp_growth)\
                    .merge(inflation)\
                    .merge(unemployment)\
                    .merge(ffr)\
                    .merge(core_inflation)\
                    .merge(nrou)\
                    .merge(potential_gdp)\


macroeconomics = macroeconomics.set_index("DATE").apply(pd.to_numeric).reset_index("DATE")

potential_gdp = potential_gdp[potential_gdp["DATE"] <= max(macroeconomics["DATE"])]
```

```python
macroeconomics.head()
```

---

### Macroeconomics Indicators and their Time Series
In this section, we will see how the value of each macroeconomic indicator varied from 1958 to present.

```python
series_names = {"GDPC1_lin": "GDP", 
               "GDPC1_pc1": "GDP Growth", 
               "CPIAUCSL_pc1": "Inflation", 
               "UNRATE_lin": "Unemployment Rate", 
               "DFF_lin": "Fed Funds Rate", 
               "CPILFESL_pc1": "Core Inflation Rate", 
               "NROU_lin": "Noncyclical Rate of Unemployment" # aka NAIRU, Natural Rate of Unemployment (Long-term)
              }

# generate a time series plot for "series"
def plot_time_series(data, series, title="1958 to present", figsize=(14, 8)):
    
    plt.figure(figsize=figsize)
    
    plt.plot(data["DATE"], data[series])
    
    # plot a horizontal line at zero if the variable is in percent
    if ("pc1" in series):
        plt.axhline(y=0, color='black', linestyle='-.')
    
    plt.xticks(data.index[np.arange(0, data.shape[0], 16)], rotation=45)
    plt.grid(visible=True)
    
    plt.xlabel("DATE")
    plt.ylabel(series)
    plt.title(f"{series_names[series]}: {title}", fontsize=16)
```

#### GDP

```python
plot_time_series(macroeconomics, "GDPC1_lin")
```

#### GDP Growth

```python
plot_time_series(macroeconomics, "GDPC1_pc1")
```

#### Inflation Rate

```python
plot_time_series(macroeconomics, "CPIAUCSL_pc1")
```

#### Core Inflation Rate

```python
plot_time_series(macroeconomics, "CPILFESL_pc1")
```

#### Unemployment Rate

```python
plot_time_series(macroeconomics, "UNRATE_lin")
```

#### Fed Funds Rate

```python
plot_time_series(macroeconomics, "DFF_lin")
```

#### Noncyclical Rate of Unemployment

```python
plot_time_series(macroeconomics, "NROU_lin")
```

What interesting pattern do you notice?

## Making an empirical Phillips curve

The above plots are interesting, but they can be found on the FRED website. Let's take the FRED data and do some data science with it.

```python
plt.scatter(macroeconomics['UNRATE_lin'], macroeconomics['CPILFESL_pc1'])
plt.title('Empirical Phillips Curve')
plt.xlabel('Unemployment Rate')
plt.ylabel('Inflation');
```

We can see a somewhat negative relationship, but it's not as strong as the relationship we saw during lecture, which used pre-WWI data. In the lab for this week, you'll use the FRED API to dive deeper into the relationship between the unemployment rate and inflation and see how it has changed over time.

```python
plt.scatter(macroeconomics['UNRATE_lin'], macroeconomics['GDPC1_lin'])
plt.title('Okun\'s Law With Emprirical Data')
plt.xlabel('Unemployment Rate')
plt.ylabel('Output (GDP)');
```

```python
plt.scatter(macroeconomics['DFF_lin'], macroeconomics['GDPC1_lin'])
plt.title('Empirical IS Curve')
plt.xlabel('Real Interest Rate')
plt.ylabel('Output (GDP)');
```

