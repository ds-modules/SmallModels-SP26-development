---
title: "Lec10.2-waterguard"
type: lecture-notebook
week: 10
source_path: "/Users/ericvandusen/Documents/Data88E-ForTraining/F24Lec_NBs/lec10/Lec10.2-waterguard.ipynb"
---

<table style="width: 100%;">
    <tr style="background-color: transparent;"><td>
        <img src="https://data-88e.github.io/assets/images/blue_text.png" width="250px" style="margin-left: 0;" />
    </td><td>
        <p style="text-align: right; font-size: 10pt;"><strong>Economic Models</strong>, Fall 2024<br>
            Dr. Eric Van Dusen</p></td></tr>
</table>

# Lec 9 : Water Guard Randomized Controlled Trial

This Lecture Notebook is an adaptation from a set of notebooks developed for a full semester Data Science Connector Course taught in Fall 2017, entitled "Behind the Curtain in Economic Development".  This dataset come from a randomized controlled trial household survey carried out in Eastern Kenya in 2007-2008. 

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
```

<!-- END QUESTION -->



<div id="Balance"></div>

## Balance Check and Variable Names

### Baseline Survey 
This is our first look at the survey dataset.  These are a limited set of questions/answers from a simple and short baseline survey. However it is a lot bigger and messier than the datasets we have seen so far and in Data 8. 

Data variable names follow along with the survey below, referred to by the section, a,b,c... number, 1,2,3... and a few words about the question. 

The purpose of this section will be: 
* to get a familiarity with the dataset, 
* to look at some background descriptor variables of the households, 
* to start to think about missing values and coding of subsets of the data.  
* to check the randomization of households by seeeing if the different arms of the study are balanced across some of the key baseline variables.  

**The surveys that illustrate the raw data names are in a file linked [here](https://drive.google.com/open?id=1UVoiVn7LJ4rn7WEb-9BJ96jmdJ2FBk60). You have to go and look through this survey to understand the variables.**

**The code sheet that has the codes for some of the possible answers are in a file linked [here](https://drive.google.com/file/d/1iinJXExeVKV4Dm7tRKOiotoYUDSXMyqc). You have to go and look through this code sheet in a later section.**

```python
WGP_baseline = Table.read_table("WGP_baseline_Data8.csv")
WGP_baseline
```

```python
baseline = pd.read_csv("WGP_baseline_Data8.csv")
baseline
```

```python
baseline.dropna(axis = 0)
```

### Misssing values 

If you look through the dataset above, and scroll to the right a ways to some of the last variables, you will notice that that there are a lot of cells with NaN, which means a missing value. For these cells no data was entered at the time of data entry. In some cases it may be appropriate to enter a zero and carry on with the analyis.

```python
WGP_base_dfna = WGP_baseline.to_df().fillna(0)
WGP_table = Table.from_df(WGP_base_dfna)
WGP_table
```

Look at the variable names, and then look at the survey form to find the concordance of codes

```python
# Here is a list of all of the possible categories / columns
list(WGP_table)
```

### What are some Variables that we want to specifically look at? ###

There are a lot of variables here and it can be kind of overwhelming, but it is good to see how many columns there can be in a comprehensive survey dataset.

#### Front Page information - A variables

- household id
- spring id
- interviewer id

#### Information about respondent - B variables 

- tribe
- education
- age
- gender 
- group membership

#### Water Guard Use - C variables

For Waterguard (WG) usage

- `c1a` - Whether the respondent has ever heard of WG
- `c2a` - Whether the respondent has ever used WG
- `c3a` - Whether the respondent's water is currently treated with WG
- `c4a` - Whether the respondent has used WG in the past month

#### Durable / Capital Goods - D variables

- Whether the respondent has electricity / latrine / iron roof
- Number of of bicycle / radio / hoe / beds owned
- Number of animals owned

#### Child Health - E variables

- `e1_num_kids_under_5`: Number of kids under 5
- `e2_`:  This table becomes tricky because it has a different format. Each kid in the table is numbered 01, 02 and so on, and then the subsequent questions are keyed to that child number. e.g. `e2e_01_d_diarrhea`, `e2e_02_d_diarrhea` represent whether child 1 and 2 respectively have diarrhea. In total, four diseases are recorded:
    - Cough
    - Diarrhea
    - Malaria
    - Vomiting

### The Treatment Arm 

In the study, arm 1 is control, while Arms 2-7 are different types of treatment interventions:
 
- Arm 1 - Control
- Arm 2 - Household Script
- Arm 3 - Community Script
- Arm 4 - HH + Community Script
- Arm 5 - Flat-Fee Promoter + Coupons
- Arm 6 - Incentivized Promoter + Coupons
- Arm 7 - Incentivized Promoter + Dispenser at Spring

Let's check how many households are in each treatment arm.

```python
WGP_table.group("treatment_arm")
```

### Baseline Check - Exposure to Water Guard Use 

Let's see how many households have ever used Water Guards.

The data is currently Coded as 1 = Yes and 2 = No, so we can't really make sense of the Mean of the variable in its current form. Instead, we will make a new column/variable with the 1 or 2 answers translated into Yes or No.
Notably, we must first filter out respondents that had missing values (with value 0) for this question.

```python
WGP_ever = WGP_table.where('c2a_wg_used_ever', are.above(0))
WGP_ever.group("c2a_wg_used_ever")
```

```python
#This helper function goes through a column of choice, and spits out yes or no based off each value in the column. It returns an array of these yes and no's
def translate_to_yesno(table, col):
    dummy=[]
    table=table.where(col, are.above(0))
    for i in np.arange(table.num_rows):
        if table.column(col).item(i) == 1:
            dummy.append('Yes')
        else: #if not 1 then its 2 and 2 means no
            dummy.append("No")
    return dummy
```

```python
new = translate_to_yesno(WGP_ever, 'c2a_wg_used_ever')
WGP_ever = WGP_ever.with_column('c2a_wg_used_ever',new)
WGP_ever.group('c2a_wg_used_ever')
```

### Pivoting and Balance Checks

Now we will use a command called **Pivot** to create a new table that has the percent of households who have ever used Water Guard within each Treatment Arm. 

We can first use it to do a  **balance check** for Water Guard use across Arms.

```python
ever_yesno = WGP_ever.pivot('c2a_wg_used_ever','treatment_arm')
ever_yesno
```

Converting to percentages...

```python
total = ever_yesno.column(1) + ever_yesno.column(2)
ever_yesno = ever_yesno.with_columns('Percent No',ever_yesno.column(1) / total * 100, 
                                     'Percent Yes', ever_yesno.column(2) / total * 100)
ever_yesno
```

Let's also repeat the process for the variable of whether the households are currently using Water Guard, `c3a_wg_water_currently_treat`.

```python
WGP_current = WGP_table.where('c3a_wg_water_currently_treat',are.not_equal_to(0))
new2 = translate_to_yesno(WGP_current,'c3a_wg_water_currently_treat')
WGP_current = WGP_current.with_column('c3a_wg_water_currently_treat',new2)
WGP_current.group("c3a_wg_water_currently_treat")
```

Do you notice a problem here? Look at the total numbers reported in the output above.

We can do the same percentage tables for the balance check but maybe there's a problem. 
Look at the total number of households answering the question and compare that to the total number from the previous section.

```python
current_yesno = WGP_current.pivot('c3a_wg_water_currently_treat','treatment_arm')
total = current_yesno.column(1) + current_yesno.column(2)
current_yesno = current_yesno.with_columns('Percent No',current_yesno.column(1)/total * 100, 
                                           'Percent Yes', current_yesno.column(2)/total * 100)
current_yesno
```

This seems like a really high usage, but **maybe this is due to missing values**. 

Let's now also include the 0 (missing) values in our analysis.

```python
current_yesnomissing = WGP_table.pivot('c3a_wg_water_currently_treat','treatment_arm')
total = current_yesnomissing.column(1) + current_yesnomissing.column(2) + current_yesnomissing.column(3)
current_yesnomissing = current_yesnomissing.with_columns(
                                     'Percent Missing',current_yesnomissing.column("0.0") / total * 100, 
                                     'Percent No',current_yesnomissing.column("2.0") / total * 100, 
                                     'Percent Yes', current_yesnomissing.column("1.0") / total * 100)
current_yesnomissing
```

<!-- END QUESTION -->



<div id="Baseline"></div>

## Baseline and a Randomly Selected Compound

Let's describe a household selected at random.

First, we will extract the household/compound id into an array.

```python
hhld_array = WGP_table.column('a1_cmpd_id')
hhld_array
```

Next, we will draw randomly from this array.

```python
randomhh = np.random.choice(hhld_array)
print("My randomly selected household is household number", randomhh)
```

Then, let's look at the data for our randomly selected household:

```python
myfamily = WGP_table.where("a1_cmpd_id",randomhh)
myfamily
```

Some of the variables may need some manipulation. 
Let's start with the age of the respondent:

```python
birthyear = myfamily.column("b3_birth_year").item(0)
surveyyear = myfamily.column("a5_date_interview_year").item(0)
agecalc = surveyyear-birthyear  # 
agecalc
```

And their tribe:

```python
print("Survey respondent Tribe", myfamily.column("b5_tribe").item(0))
print("Respondent Spouse Tribe", myfamily.column("b7_tribe_spouse").item(0))
```

Lastly, whether they have a latrine:

```python
print("Does the household have a latrine?", myfamily.column("d3_latrine").item(0))
```

Remember in the answer above it is coded so that 1=Yes and 2=No.

<!-- BEGIN QUESTION -->

**Question 3:** Describe your randomly selected household and the respondent who is answering the survey. Please remember you can find the code sheet under the section of Baseline Survey.

1. Age
2. Tribe
3. Education 
4. Member of any groups b11-b15?
5. Occupation
6. Religion 
7. A summary of D variables, iron roof, floor materials, latrine, cattle, and others
8. Have they ever used WG?
9. Their treatment arm assignment
10. How many children do they have  
11. Gender and Age of children
12. Have any of the children been sick?

<!--
BEGIN QUESTION
name: q3
manual: true
-->

_Type your answer here, replacing this text._

<!-- END QUESTION -->



<div id="Chlorine"></div>

## Water Guard Usage outcome variables

### WGP Followup - Variability
The purpose of this section will be to continue on with the follow-up rounds of the Water Guard Promotion study.   In this section we have both the household reported use, and the use validated by checking the chlorine content of the water using a test kit.

```python
WGP3rds_table = Table.read_table('WGP_3waves_Data8.csv')
WGP3rds_table
```

This is a large dataset, basically three datasets merged together, one for baseline, one for short term follow up and one for long term followup. The column `round` describes these 3 time steps:

- Round = 1 : baseline
- Round = 2 : 3 week followup
- Round = 3 : 3 month followup

Notably, many of the variables are only asked in one of the three rounds. For example, the chlorine use variables are:

- The variable for self reported chlorine use was `c6n` in Round 2, and `c5n` in Round 3.
- The variable for chlorine use is `c12n21pnk` in Round 2 and `c15npt2or1pnk` in Round 3.

Instead, the following variables have been combined across rounds for the ease of programming:

- `Selfrptpct` is self reported chlorine use in both round 2 and round 3
- `Vldclpct`  is validated chlorine use in both rounds

```python
WGP3rds_table.group("treatment_arm")
```

```python
WGP3rds_table.group('round')
```

### Grouping by round + treatment arm

We want to create a multi-level group: each group should be a unique combination of the survey round and the treatment arm.

```python
WGP_3rds_outcomesonly= WGP3rds_table.select("round", "treatment_arm", "Selfrptpct", "Vldclpct")
WGP_3rds_outcomesonly.group(["round","treatment_arm"], np.mean).show(30)
```

### Making a smaller dataset

Lets break out a smaller dataset of the variables we want to focus on; just for Round 2 and the outcome variables.

```python
WGPRd2 = WGP3rds_table.where("round", 2).select("a1_cmpd_id","treatment_arm",
                                           "c6_current_water_treated_wg", 
                                           'c6_curr_water_treat_other_c',
                                           'c12_chlorine_meter_reading',
                                           'c11_chlorine_color','c12n21pnk', 'c6n'
                                          )
WGPRd2
```

A quick examination of the estimated Water Guard usage in Round 2 across all treatment arms:

```python
np.mean(WGPRd2.column('c12n21pnk'))
```

```python

```

```python

```

