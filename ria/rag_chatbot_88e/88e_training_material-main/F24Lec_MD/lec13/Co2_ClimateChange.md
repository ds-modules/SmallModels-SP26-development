---
title: "Co2_ClimateChange"
type: lecture-notebook
week: 13
source_path: "/Users/ericvandusen/Documents/Data88E-ForTraining/F24Lec_NBs/lec13/Co2_ClimateChange.ipynb"
---

## Carbon Dioxide Emissions and Global Climate Change

This notebook is a very simple analysis of global carbon emissons and global temperature change.

- The first point is to load the data directly from the government agencies that manage it.   
- The second is to plot the data and see that we are at the hightest levels of carbon emissions in history.  
- The third is to plot the data and see that we are at the highest levels of global temperature in history.  
- The fourth is to plot the two data series and see that there is a correlation between the two.

```python
import pandas as pd
import plotly.express as px
```

## Global Atmospheric Carbon Levels

The CO2 data is from the Global Monitoring Laboratory of the National Oceanic and Atmospheric Administration (NOAA).  The most frequently cited data is the Mauna Loa Observatory in Hawaii.  

![NOAA](https://upload.wikimedia.org/wikipedia/commons/1/11/NOAA_logo_mobile.svg)

The data is available at https://www.esrl.noaa.gov/gmd/ccgg/trends/data.html.

There are many data series, but we have selected monthly mean data, which are available as a csv.  

Mauna Loa Obervatory is at the top of a volcano in the middle of the Pacific Ocean.  It is a good place to measure CO2 because it is far from any major sources of CO2.  The data is available from 1958 to the present.  The data is in parts per million (ppm) of CO2 in the atmosphere.

```python
url_mlo_co2="https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"

df_mlo_co2 = pd.read_csv(url_mlo_co2, comment='#')

df_mlo_co2
```

### Let's graph the monthly level of Co2 over time to see the trend.

This is potentially one of the most famous graphs in the world.  It is called the Keeling Curve, after Charles David Keeling, who started the measurements in 1958.  The graph shows the monthly mean level of CO2 in the atmosphere.

```python


fig = px.line(df_mlo_co2, x='decimal date', y='average', title='Mauna Loa CO₂ daily means')
fig.show()
```

## GISS Surface Temperature Analysis (GISTEMP v4) ##

This dataset is the GISS Surface Temperature Analysis (GISTEMP v4) provided by NASA's Goddard Institute for Space Studies. The dataset is updated on a monthly basis and is available from 1880 to the present. The dataset is a combination of land-surface air temperatures and sea-surface water temperatures. The data is provided in tabular format and is available for global, hemispheric, and zonal means. The data is provided as temperature anomalies, which are deviations from the 1951-1980 means. The data is available in csv files and is updated through the most recent month.

![NASA](https://www.giss.nasa.gov/gfx/2013/nasa_x2.png)

The data is available from the following URL:
https://data.giss.nasa.gov/gistemp/

We have chosen the following datasets for this analysis:

**Global-mean monthly, seasonal, and annual means, 1880-present, updated through most recent month**

```python
url_giss_temp = 'https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv'

giss_temp = pd.read_csv(url_giss_temp, skiprows=1)

giss_temp
```

Lets plot the data to see the trend in global temperature over time.

MAM is the variable name for Monthly Anomaly Means.  The data is in degrees Celsius.

```python
fig = px.line(giss_temp, x='Year', y='MAM', title='Global temperature anomaly')
fig.show()
```

## Correlation between CO2 and Global Temperature ##

Let's check the correlation between the two datasets.  

First we need to cut the temperatture anomoly data to match the CO2 data.  The CO2 data is from 1958 to the present.  The temperature data is from 1880 to the present.  We will cut the temperature data to match the CO2 data.  

Then we will make an annual CO2 dataset by taking the average of the monthly data.

Finally we will merge the two datasets on the year.

```python
# cut temp  data to 1958 
giss_temp = giss_temp[giss_temp['Year'] >= 1958]
```

```python
# make an annual graph for the Co2 data
#df_mlo_co2['year'] = df_mlo_co2['decimal date'].apply(lambda x: int(x))
annual_co2 = df_mlo_co2.groupby('year').mean().reset_index()
annual_co2
```

```python
#merge with co2 data
merged_co2_temp = pd.merge(annual_co2, giss_temp, left_on='year', right_on='Year')
```

```python
merged_co2_temp
```

```python
#graph over time using plotly for the merged data
fig = px.scatter(merged_co2_temp, x='average', y='MAM', 
    title='CO₂ vs global temperature anomaly',
    labels={'average':'CO₂ (ppm)', 'MAM':'Temperature anomaly (°C)'},
    trendline='ols')
# add a trendline


fig.show()
```

```python
#calculate the correlation
correlation = merged_co2_temp['average'].corr(merged_co2_temp['MAM'])
print(correlation)
```





```python

```

