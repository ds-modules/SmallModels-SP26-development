---
title: "EmissionsTracker"
type: lecture-notebook
week: 13
source_path: "/Users/ericvandusen/Documents/Data88E-ForTraining/F24Lec_NBs/lec13/EmissionsTracker.ipynb"
---

## Climate Action Tracker - Country Assessments 


The Climate Action Tracker develops national greenhouse gas emission scenarios to assess countries' mitigation efforts. 
These scenarios include both pledges, such as NDCs and net zero commitments, and emission trajectories under currently adopted policies and actions.
For more details please visit the methodology section on our website: https://climateactiontracker.org/methodology/estimating-national-emissions/

Here, we present the latest version of the scenarios developed for each country. Please note that the assessment date varies per country and that some updates from the past month may not yet be reflected in this table.

Values in the table below are in MtCO2e/year and AR4 GWPs.

Please reference as: 'Climate Action Tracker, Country Assessments | November 2024 - http://climateactiontracker.org'

Copyright © 2024 Climate Action Tracker by NewClimate Institute and Climate Analytics. All rights reserved. The content provided by this website is protected by copyright. 
You are authorised to view, download, print and distribute the content from this website subject to the following condition: Any reproduction, in full or in part, must credit Climate Analytics and NewClimate Institute and must include a copyright notice.

```python
import pandas as pd
import openpyxl
import plotly.express as px
```

```python
url='https://climateactiontracker.org/documents/1289/CAT_14112024_CountryAssessmentData_DataExplorer.xlsx'
```

```python
file_path = 'CAT_14112024_CountryAssessmentData_DataExplorer.xlsx'
sheet_name = 'Assessment'
```

```python
# Reload the data with the first row as headers using openpyxl engine
emissions = pd.read_excel(url, sheet_name=sheet_name, skiprows=18, engine='openpyxl')
emissions
```

LULUCF stands for Land Use, Land-Use Change, and Forestry. In emissions tracking, it refers to the greenhouse gas (GHG) emissions and removals resulting from human activities in forests, croplands, grasslands, wetlands, and other land-use types.

LULUCF plays a dual role in emissions accounting:

Source of Emissions: When forests are cleared, burned, or degraded, or when land is converted to agricultural or other uses, it releases stored carbon into the atmosphere.

Carbon Sink: LULUCF activities can also act as a carbon sink by removing CO₂ from the atmosphere, for example, through reforestation, afforestation, or improved land management practices.

In climate agreements and national emissions inventories, LULUCF is often included separately due to its unique nature of both emitting and absorbing

```python
emissions_cleaned = emissions.dropna(subset=["Country"])
emissions_cleaned
# i want to replace the value in the column "Sector" with "Emissions-exLU" where the value is "Economy-wide, excluding LULUCF"
emissions_cleaned['Sector'] = emissions_cleaned['Sector'].replace("Economy-wide, excluding LULUCF", "Emissions-exLU")
#emissions_cleaned.loc[emissions_cleaned['Sector'] == "Economy-wide, excluding LULUCF", 'Sector'] = "Emissions-exLU"

emissions_cleaned
```

```python
# lets look at China
china = emissions_cleaned[emissions_cleaned['Country'] == 'CHN']
china
```

### Lets looks at China Historical Emissions

Scenario = "Historical" and sector = "Emissions-exLU"

```python
china_historical = china[(china['Scenario'] == 'Historical') & (china['Sector'] == 'Emissions-exLU')]
china_historical
```

###  The data are in the wrong shape for analysis.
- The years are spread out as columns
- Let's  `melt` the DataFrame so that years become rows under a 'Year' column

```python
china_historical_melted = pd.melt(
    china_historical, 
    id_vars=['Version', 'Country', 'Scenario', 'Sector', 'Indicator', 'Unit'],  
    var_name='Year',
    value_name='Emissions'  
)

china_historical_melted['Year'] = pd.to_numeric(china_historical_melted['Year'], errors='coerce')

china_historical_melted.head()
```

```python
fig = px.line(china_historical_melted, x='Year', y='Emissions', title='China Emissions-exLU Historical')
fig.show()
```

##  Now let's get the Forecast data 

In this dataset these are called "current policy scenario,max" and "current policy scenario,min".  
- We will use the same process 'melt' as above to get the data in the right shape for analysis.

```python
china_current_policy = china[china['Scenario'].str.contains('Current Policy')]

china_current_policy_melted = pd.melt(
    china_current_policy_both, 
    id_vars=['Version', 'Country', 'Scenario', 'Sector', 'Indicator', 'Unit'], 
    var_name='Year',  
    value_name='Emissions' 
)

china_current_policy_melted
```

Now we need to pivot the data so that we have the following columns:
- Year
- Max Emissions
- Min Emissions

```python
data_melted = china_current_policy_melted
china_current_policy_melted['Year'] = pd.to_numeric(china_current_policy_melted['Year'], errors='coerce')

china_forecast_pivot = china_current_policy_melted.pivot_table(
    index=['Version', 'Country', 'Sector', 'Indicator', 'Unit', 'Year'],  
    columns='Scenario',  
    values='Emissions'  
).reset_index()  

# Rename the columns for clarity
china_forecast_pivot.columns.name = None  
china_forecast_pivot.rename(columns={'Current Policy, Max': 'Max_Forecast', 'Current Policy, Min': 'Min_Forecast'}, inplace=True)

china_forecast_pivot
```

Let's plot the data to see the trends in the emissions

```python
fig = px.line(china_forecast_pivot, x='Year', y='Max_Forecast', title='China Emissions-exLU Current Policy')
fig.add_scatter(x=china_forecast_pivot['Year'], y=china_forecast_pivot['Min_Forecast'], mode='lines', name='Forecast Min')
fig.show()
```

Let's add historical data to the plot

```python
fig = px.line(china_historical_melted, x='Year', y='Emissions', title='China Emissions-exLU Historical')
fig.add_scatter(x=china_forecast_pivot['Year'], y=china_forecast_pivot['Max_Forecast'], mode='lines', name='Max Emissions')
fig.add_scatter(x=china_forecast_pivot['Year'], y=china_forecast_pivot['Min_Forecast'], mode='lines', name='Min Emissions')
fig.show()
```

```python

```

```python

```

```python

```

