---
title: "RoslingPlots"
type: lecture-notebook
week: 13
source_path: "/Users/ericvandusen/Documents/Data88E-ForTraining/F24Lec_NBs/lec13/RoslingPlots.ipynb"
---

<table style="width: 100%;">
    <tr style="background-color: transparent;"><td>
        <img src="https://data-88e.github.io/assets/images/blue_text.png" width="250px" style="margin-left: 0;" />
    </td><td>
        <p style="text-align: right; font-size: 10pt;"><strong>Economic Models</strong>, Fall 2024<br>
            Peter Grinde-Hollevik<br>
            Dr. Eric Van Dusen
        </p></td></tr>
</table>

```python
from datascience import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
%matplotlib inline 
from matplotlib.pyplot import figure
pd.plotting.register_matplotlib_converters()
import seaborn as sns
from pylab import *
from ipywidgets import interact
from IPython.lib.display import YouTubeVideo
```

### Plotting Global Emissions with "Rosling"-style Plots

This notebook is inspired by the late Professor Hans Rösling's wonderful contributions to the world of data visualization, which can be seen at the Gapminder website. 

[You can see the Gapminder version of this visualization here](https://www.gapminder.org/tools/#$model$markers$bubble$encoding$y$data$concept=co2_emissions_tonnes_per_person&space@=country&=time;;&scale$domain:null&zoomed:null&type:null;;&frame$value=2018;;;;;&chart-type=bubbles&url=v1).  

This notebook aims to reproduce the "bubble charts" in understanding the historical, global emissions of the last century.

As a warm-up exercise, feel free to check out his famous ["200 Countries, 200 years, 4 minutes"](https://www.youtube.com/watch?v=jbkSRLYSojo) from the BBC series *The Joy of Stats*. This video plots life expectancy over changing per capita income. After watching this, you'll understand why the arguments made by Professor Rösling's visualization have had such an impact.

```python
YouTubeVideo('jbkSRLYSojo')
```

### Building Our Own Rosling Plot

The goal of this exercise is to gain some experience in data cleaning, making sense of it with basic plots, and present it in the most "friendly" manner for scientific communication. The two first will be accomplished by using tools from DATA 8, while the latter requires the usage of methods you might not be familiar with yet. However, with this walkthrough, you ought to be more comfortable with building a Rosling Plot of your own in the future.

We start off by importing a Gapminder Foundation Dataset containing per capita GDP and per capita emissions of $CO_2$.

```python
co2_table = Table.read_table('co2-emissions-vs-gdp.csv')
co2_table
```

Our first Rosling Plot will focus on the latest year of data: 2018. Hence, we select only the rows where 'Year' = 2018.

```python
co2_table = co2_table.where('Year', 2018)
co2_table
```

Now, let's consider what we would like to plot for year 2018: Following Rosling's philosophy we should consider how emissions change over economic development. Hence, our x-axis should be GDP per capita. Our y-axis could be a new column: Total Emissions, made by the product of Per Capita $CO_2$ emissions and the total population of each country.

```python
#Make the 'Total CO2 Emissions' column
total_co2_emissions = co2_table.column('Per capita CO2 emissions') * co2_table.column('Total population (Gapminder, HYDE & UN)')
co2_table = co2_table.with_column('Total CO2 Emissions', total_co2_emissions )
co2_table.sort("GDP per capita", descending=True)
```

Observe the 'nan' values spread across almost all columns. These are 'empty' data points, where data has simply not been collected. To ensure the validity of our analysis, we decide to remove all rows with 'per capita $CO_2$ emissions' and 'total $CO_2$ emissions' are empty. For this, we use the **.where** and **.are.above(0)** methods from DATA 8. Furthermore, we select the columns relevant to our inquiry. We also sort the table to have the biggest emitters at the top and remove any continent from the 'entity column'.

```python
#Selecting relevant columns, removing 'NaN' values, & sorting descending
co2_table = co2_table.select('Entity', 'Total CO2 Emissions', 'GDP per capita')
co2_table = co2_table.sort('Total CO2 Emissions', descending=True)
co2_table = co2_table.where('Total CO2 Emissions', are.above(0))
co2_table = co2_table.where('GDP per capita', are.above(0))
continents = make_array('World', 'Asia', 'Europe', 'Africa', 'North America', 'Latin America', 'Oceania')
co2_table = co2_table.where('Entity', are.not_contained_in(continents))
co2_table
```

Having cleaned our data, look at a scatter plot and see if we can make sense of it. What might we expect to see plotting Total CO2 emissions over GDP per capita? Do you expect any outliers? If so, what countries do you think that would be?

```python
#Let's scatter plot our data. See any association between GDP per capita and Total CO2 Emissions?
co2_table.scatter('GDP per capita', 'Total CO2 Emissions')
```

This plot is hard to read, and is dominated by a few outliers (large emitters and very wealthy nations). Hence, let us take the log of each axis. Percentage changes in GDP per capita should now refer to percentage changes in total emissions in each country.

```python
# use np.log on each column
gdp_log = np.log(co2_table.column('GDP per capita'))
co2_log = np.log(co2_table.column('Total CO2 Emissions'))

#Log-plot. Easier to see the association?
log_co2_table = co2_table.with_columns('Log GDP per capita', gdp_log, 'Log Total CO2 Emissions', co2_log)
log_co2_table.scatter('Log GDP per capita', 'Log Total CO2 Emissions', fit_line = True)
```

To find out the slope and intercept of that line fitted through the data, we run a simple Linear Regression model using Numpy.

```python
x = gdp_log
y = co2_log
np.polyfit(x, y,1)
```

### Part 2: A Historical View on $CO_2$ emissions

As of now, it seems like we have a strong linear association between log total $CO_2$ emissions and log GDP per capita. To explore this association further, let us view this from a historical perspective by answering the following question: How has the relationship between the two variables changed over time? To do so, we build the Rosling Plot we sought out in the first place; one for each year the last 120 years.

```python
#For this, we need the population & continent of each country.
gapminder = Table.read_table('gapminder - gapminder.csv')
gapminder
```

We use the DATA 8 method **.join** you are familiar with by now to add continent and population for each nation. The color of the "bubbles" we are making will represent the continent, while its size represents the population.

```python
bubble_table = co2_table.join('Entity', gapminder, 'country')
bubble_table = bubble_table.select('Entity', 'Total CO2 Emissions', 'GDP per capita','population','continent')
bubble_table.sort('Total CO2 Emissions', descending=True)
```

Using this newly created table, we can visualize the log-plot from above in a "friendlier" manner using an imported method called Seaborn. Feel free to check out its documentation here: https://seaborn.pydata.org/generated/seaborn.scatterplot.html

```python
population = bubble_table.column('population')
plt.figure(figsize = (8,6), dpi=250)
#sns.scatterplot(bubble_table.column('GDP per capita'), bubble_table.column('Total CO2 Emissions'), hue = bubble_table.column('continent'), size= population, sizes=(20,400), legend=False);
sns.scatterplot(
    x=bubble_table.column('GDP per capita'), 
    y=bubble_table.column('Total CO2 Emissions'), 
    hue=bubble_table.column('continent'), 
    size=population, 
    sizes=(20, 400), 
    legend=False
)
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('GDP per capita [USD]')
plt.ylabel('Total CO2 Emissions [Tons]')
plt.title('World CO2 Emission in 2018')
plt.xticks([1000, 10000, 100000],['1k', '10k', '100k'])
plt.show();
```

So, this was our Rosling Plot for 2018. From here, we aim to create a function that we can apply to each year in our original dataset. Remember the one with 50.000+ entries, each representing a country in a given year?

One thing to note is that our could be changed a tad: Would it not make sense to plot $CO_2$ emissions per capita over GDP per capita? Doing so, we get a sense of the $CO_2$ intensity of GDP.

```python
def the_bubble_plot(emissions_data, population_data, income_data, geography_data, year):
    # Selecting data for given year & relabel columns
    emissions_data = emissions_data.select('country', f"{year}").relabel(1, 'CO2 per capita [tons]')
    population_data = population_data.select('country', f"{year}").relabel(1, 'Population')
    income_data = income_data.select('country', f"{year}").relabel(1, 'GDP per capita [USD]')

    # Creating the 'the_bubble_table' with emission, population, income, continent for given year
    the_bubble_table = emissions_data.join('country', population_data)
    the_bubble_table = the_bubble_table.join('country', income_data)
    the_bubble_table = the_bubble_table.join('country', geography_data)

    # Filter to ensure we have valid emissions data for the given year
    the_bubble_table = the_bubble_table.where('CO2 per capita [tons]', are.above(0))

    # Prepare data for Seaborn (convert to NumPy arrays)
    gdp_per_capita = the_bubble_table.column('GDP per capita [USD]')
    co2_per_capita = the_bubble_table.column('CO2 per capita [tons]')
    population = the_bubble_table.column('Population')
    continent = the_bubble_table.column('continent')

    # Create the scatter plot
    plt.figure(figsize=(8, 6), dpi=250)
    sns.scatterplot(
        x=gdp_per_capita,
        y=co2_per_capita,
        hue=continent,
        size=population,
        sizes=(20, 400),
        legend=False
    )
    
    # Adjusting the plot for better visualization
    plt.grid(False)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('GDP per capita [USD]', fontsize='large')
    plt.ylabel('CO2 per capita [tons]', fontsize='large')
    plt.title(f"World CO2 Emissions in {year}", fontsize='x-large')
    plt.xticks([10**0, 10**1, 10**2], ['1k', '10k', '100k'])
    plt.yticks([10**-2, 10**-1, 10**0, 10**1], ['0.01', '0.1', '1', '10'])
    plt.show()


# Importing data from the Gapminder Foundation
emissions_data = Table.read_table('co2_emissions_tonnes_per_person.csv')
population_data = Table.read_table('population_total.csv')
income_data = Table.read_table('income_per_person_gdppercapita_ppp_inflation_adjusted.csv')
geography_data = Table.read_table('gapminder - gapminder.csv').select('country', 'continent')

# Call the function for the year 2017
the_bubble_plot(emissions_data, population_data, income_data, geography_data, 2017)
```

```python

```

