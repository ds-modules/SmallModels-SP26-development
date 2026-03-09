---
title: "lec10.1-mapping"
type: lecture-notebook
week: 10
source_path: "/Users/ericvandusen/Documents/Data88E-ForTraining/F24Lec_NBs/lec10/lec10.1-mapping.ipynb"
---

<table style="width: 100%;">
    <tr style="background-color: transparent;"><td>
        <img src="https://data-88e.github.io/assets/images/blue_text.png" width="250px" style="margin-left: 0;" />
    </td><td>
        <p style="text-align: right; font-size: 10pt;"><strong>Economic Models</strong>, Fall 2024
        <br>
            Dr. Eric Van Dusen</p></td></tr>
</table>

# Lec9: Water Guard Randomized Controlled Trial

This notebook is an adaptation from a set of notebooks developed for a full semester Data Science Connector Course taught in Fall 2017, entitled "Behind the Curtain in Economic Development".  This dataset come from a randomized controlled trial household survey carried out in Eastern Kenya in 2007-2008. 

The purpose of the study was to understand how to promote the use of WaterGuard, a dilute sodium hypochlorite solution that was promoted for Point-of-use household water disinfection.  There were seven arms in the study, which will be more fully described in the following chart:

<img src="Slide1.png"  />

Within this table you can see the seven treatments arms -  control plus three treatments -  in the bolded boxes in the middle with the number of springs and households. The study was carried out as a part of a study of households who gather drinking water from springs in a rural area.  The three boxes at the bottom describe the three rounds of data collection - a baseline before the treatment, and a short term and long term follow-up.

<!-- **Notebook Outline**

1. [Mapping](#Mapping)
2. [Balance Check](#Balance)
3. [Baseline and a Randomly Selected Compound](#Baseline)
4. [Chlorine Usage outcome variables](#Chlorine)
5. [Graph of outcomes by Treatment Arm](#Graph)  -->

```python
from datascience import *
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
from pandas import read_stata
from ipyleaflet import Map, basemaps, Marker, AwesomeIcon
```

## Mapping

<div id="Mapping"></div>

This first section works with a package in Jupyter called ipyleaflet.

`ipyleaflet`;
the documentation is [here](https://ipyleaflet.readthedocs.io/en/latest/)
and it is worth a short read through if you are interested.


We want to use two different base maps - one is a satellite layer and oen is the Open Street Map layer.

We will start by reading in a dataset of the coordinates of the springs that are used in the WaterGuard Promotion (WGP) study.  These springs were randomized into seven different treatment arms.  The springs are identified by a unique numerical id tag, and the common name in the local language.

```python
springsGPS = Table.read_table('WGPgps_forData8.csv')
springsGPS
```

```python
# make a table wth just the North and East Gps columns 
locations = springsGPS.select("gpsn1", "gpse1")
locations
```

Where in the world are we?

First of all lets look at the mean for the Lat and Long and we can center our map there

```python

mean_longitude = springsGPS.column('gpse1').mean()
mean_latitude = springsGPS.column('gpsn1').mean()

print("Mean of 'gpse1':", mean_longitude)
print("Mean of 'gpsn1':", mean_latitude)
```

The code cell below should display a map. However, it may not run the first time you click it - if this happens, try running all the cells above this one and then refreshing your browser. After a few refreshes, the maps should load.

```python

center = [0.4, 34.4]
zoom = 12
basemap=basemaps.Esri.WorldImagery
layout={'width': '800px', 'height': '600px'}

Map(basemap=basemap, center=center, zoom=zoom, layout=layout)
```

Lets make a map of our sample sites ( springs)

```python
m = Map(basemap=basemap, center=center, zoom=zoom, layout=layout)

# Iterate through the rows in the dataset
for row in springsGPS.rows:
    latitude = row.item('gpsn1')
    longitude = row.item('gpse1')
    marker = Marker(location=(latitude, longitude))
    m.add_layer(marker)

m
```

Now the most interesting bit of data is still not being used, the Treatment Arm. Let's assign different colors to the different treatment arms so that when we map it we can see if the arms appear to be randomly distributed.

The following is function assigns the 7 different treatment arms to a set of colors. [Here](https://www.w3.org/TR/css3-color/#html4) is the colors reference if you are interested!

```python
def color(arm):
    if arm == 1:
        return 'black'
    elif arm == 2:
        return 'red'
    elif arm == 3:
        return 'purple'
    elif arm == 4:
        return 'green'
    elif arm == 5:
        return 'blue'
    elif arm == 6:
        return 'pink'
    elif arm == 7:
        return 'orange'
```

```python
# Using the .apply method, you can apply any function to a data frame
colors = springsGPS.apply(color, "treatment_arm")
springsGPS = springsGPS.with_column("color", colors)
springsGPS
```

```python

m = Map( center=center, zoom=zoom, layout=layout)

for row in springsGPS.rows:
    latitude = row.item('gpsn1')
    longitude = row.item('gpse1')
    color = row.item('color')
    
    marker = Marker(
        location=(latitude, longitude),
        draggable=False,  # Set to True if you want to make the markers draggable
        title=color,      # Set the marker title to the color for tooltip
        alt=color         # Set the alt text to the color
    )
    
    # Apply the specified color to the marker
    marker.icon = AwesomeIcon(name='circle', marker_color=color)
    
    m.add_layer(marker)

m
```

```python

m=Map(basemap=basemap, center=center, zoom=zoom, layout=layout)

for row in springsGPS.rows:
    latitude = row.item('gpsn1')
    longitude = row.item('gpse1')
    color = row.item('color')
    
    marker = Marker(
        location=(latitude, longitude),
        draggable=False,  # Set to True if you want to make the markers draggable
        title=color,      # Set the marker title to the color for tooltip
        alt=color         # Set the alt text to the color
    )
    
    marker.icon = AwesomeIcon(name='circle', marker_color=color)
    
    m.add_layer(marker)

m
```

Do the colors seem randomly distributed?

In fact, the randomization was performed on just a list of the springs using a random number generator. 
It did not take spatial distribution into effect.

```python

```

```python

```

