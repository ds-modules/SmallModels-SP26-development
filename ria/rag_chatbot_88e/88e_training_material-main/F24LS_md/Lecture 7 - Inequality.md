---
title: "Lecture 7 - Inequality"
type: slides
week: 7
source_path: "/Users/ericvandusen/Documents/Data88E-ForTraining/F24LS/Lecture 7 - Inequality.pptx"
---

## Slide 1: Data 88E: Economic Models

- Lecture 7: Inequality

## Slide 2: Announcements

- Assignments
  - Project 2 underway Use Ed megathreads and office hours if you have questions.
  - Lab 6 will be released, it’ll be due before next class

## Slide 3: What about A - Total Factor Productivity

- Lab 4 and Proj 2 talk about A
- A theoretical Model of a production - a term for productivity
- And outcome of an estimation ( Intercept term)
- Growth economics is a whole field of Macroeconomics
- A should grow over time ( accumulation of productivity)

## Slide 4: Lab 5 three models of Utility

- Mathematical formulations to represent  assumptions about Utility
- Complements - Utility comes from combination
- Substitutes - Utility comes from either good equally
- Eg Cobb Douglas - somewhere in between

## Slide 5: Lab 5 postscript

## Slide 6: Agenda

- Why Inequality
- Measuring Inequality
  - Lorenz Curve
  - Gini Coefficient
- Income Inequality Historically
- Income Inequality Internationally
- Textbook Chapter has code for most visualizations in class today
- Data Visualizations - From Econ Profs Saez and Zucman
- Let’s explore these together
- We have a few website data demos - find them either through the website or through opening this slide deck and following the links

## Slide 7: Why income inequality matters

- Further reading

<details><summary>Speaker notes</summary>

P

</details>

## Slide 8: Alesina and Perotti - 1996

## Slide 9: How does income inequality look like?

- Dollar Street Demo
- https://www.gapminder.org/dollar-street
- 458 families in 66 countries, with 43971 photos
- Go to Dollar Street and find one the following:
- How do people cook?
- Live?
- Brush their teeth?
- Wash hands?
- Worship?
- Dream of having?
- Talk in pairs:
- How is a photo different than a data point

<details><summary>Speaker notes</summary>

P

</details>

## Slide 10: (untitled)

## Slide 11: Income vs Wealth

- Income - Amount of money you make during a specified time frame
  - This is what we’ll focus on today
  - All of our analysis today will be pre-tax!
- Wealth - total value of assets - liabilities
  - Assets: real estate, securities, cash
    - Luxuries: Antiques
    - Intangible: patents, copyrights, etc
  - Liabilities: mortgages, loans, etc

<details><summary>Speaker notes</summary>

E

</details>

## Slide 12: Measuring Income Inequality

<details><summary>Speaker notes</summary>

E

</details>

## Slide 13: Lorenz Curve

- Rank every person in order by how much income
- ( rank each bin of the population - eg decile = 10%)
- X axis - income percentile
- Y axis -  at the Xth percentile, share of combined income for everyone below the Xth percentile
- For any point (x,y) on the Lorenz curve, we can say that “the bottom x percent own y% of the income”.

<details><summary>Speaker notes</summary>

E

</details>

## Slide 14: An example

<details><summary>Speaker notes</summary>

E

</details>

## Slide 15: An example

<details><summary>Speaker notes</summary>

E

</details>

## Slide 16: An example

- Just percentiles
- From the data

<details><summary>Speaker notes</summary>

E

</details>

## Slide 17: An example

<details><summary>Speaker notes</summary>

E

</details>

## Slide 18: Conditions of the Lorenz Curve

- (0,0) and (1,1) on the curve
  - (0,0): 0% of the households own 0% of the total income
  - (1,1): 100% of the households own 100% of the total income
- Slope is always increasing (convex function, second derivative > 0)
  - X-axis is based on income percentile, which sorts households in ascending order of income
  - E.g. the household at the 1st percentile will earn less than the household at the 2nd percentile. Hence, the change in total income % at the 2nd percentile must be greater than the change in total income at the 1st percentile
  - We can apply this logic to every percentile!

<details><summary>Speaker notes</summary>

E

</details>

## Slide 19: Line of Perfect Equality

- What would the Lorenz curve look like if our economy was perfectly equal?
- Hint: everyone has the same income!

<details><summary>Speaker notes</summary>

E

</details>

## Slide 20: Line of Perfect Equality

- Since everyone has the same income:
- 1st percentile & under owns 1% of income
- 2nd percentile & under owns 2% of income
- …
- 100th percentile & under owns 100% of income

<details><summary>Speaker notes</summary>

E

</details>

## Slide 21: Which country has more inequality?

<details><summary>Speaker notes</summary>

E

</details>

## Slide 22: Which country has more inequality?

- Bottom 10% own:
  - 2% in Country1
  - 3% in Country2
- Bottom 50% own:
  - 23% in Country1
  - 27% in Country2
- Top 10% own:
  - 100-75=25% in Country1
  - 100-81=19% in Country2

<details><summary>Speaker notes</summary>

E

</details>

## Slide 23: Which country has more inequality?

## Slide 24: Which country has more inequality? (II)

## Slide 25: Which country has more inequality? (II)

- Bottom 10% own:
  - 2% in Country1
  - 3% in Country3
- Bottom 50% own:
  - 23% in Country1
  - 25% in Country3
- Top 10% own:
  - 100-75=25% in Country1
  - 100-73=27% in Country3

## Slide 26: Which country has more inequality? (II)

## Slide 27: Motivation for the Gini Coefficient

- The previous example highlights an ambiguity with directly using the Lorenz Curve as a measure of income inequality.
- To address this, we turn to the Gini Coefficient

## Slide 28: Gini Coefficient

## Slide 29: Gini Coefficient - Using Calculus

## Slide 30: Gini Coefficient - Using Calculus

- In the homework, you will empirically apply a Lorenz curve function to different countries, and use that to compute the Gini coefficient!

## Slide 31: Gini Coefficient - Intuition

- What is the Gini coefficient if the economy is perfectly equal?
- What is the Gini coefficient if the economy is completely unequal?

## Slide 32: Gini Coefficient - Intuition

- What is the Gini coefficient if the economy is perfectly equal?
- Since Lorenz curve = Line of equality, A = 0. Hence, Gini = 0.
- What is the Gini coefficient if the economy is completely unequal?
- We can imagine that complete inequality entails 1 person owning all the income. Hence, The area of B is minimized to be essentially 0. Hence, Gini = 1.

## Slide 33: An Aside - Lab 6

- https://www.tandfonline.com/doi/ref/10.1080/02664768700000032

## Slide 34: Lab 6 - and Sympy can do Integration as well!

## Slide 35: Gini Coefficients Around the World

- Sweden: 0.27
- India: 0.35
- Singapore: 0.46
- Canada: 0.34
- Kenya: 0.48
- United States: 0.42
- Most Equal: Liberia, Afghanistan, Iraq
- Most unequal: South Africa, Haiti, Botswana, Namibia
- OECD Data

## Slide 36: OECD table -

- https://data.oecd.org/inequality/income-inequality.htm

## Slide 37: Summary of the Gini index

- The Gini coefficient is a good measure for the magnitude of income inequality, allowing us to easily compare income inequality across countries.
- Typically, Gini coefficients range between 0.25 and 0.5
- Computing the Gini coefficient is quite complicated: comprehensive income data may not always be collected
- What about wealth inequality?

## Slide 38: Lecture Notebook 1

## Slide 39: Income Inequality Historically

## Slide 40: Factors that Impact Income Inequality

- Key factors and alternative causes
  - Top Marginal Tax Rates
  - Unemployment Rates
  - Population Growth
- Trends
- Policy implications
- Notebook for the following slides is the Notebook for the textbook chapter!

## Slide 41: Historically - Lorenz Gini can be hard to come by

- Simplify
- Percent of Income to bottom 50%
- Percent of Income to top 10%
- Percent of Income to top 1%

## Slide 42: US Income Inequality (Raw Data)

- Percentile
  - Range of income bracket
- Income share
  - Some have values, some are nan
  - Why are some nan?

## Slide 43: US Income Inequality (Raw Data)

- Percentile
  - Range of income bracket
- Income share
  - Some have values, some are nan
  - Why are some nan?
    - Data wasn’t being collected

## Slide 44: US Income Inequality (Cleaned Data)

## Slide 45: US Income Inequality Graphed

## Slide 46: (untitled)

## Slide 47: (untitled)

## Slide 48: (untitled)

## Slide 49: Top Marginal Tax Rates

- Tax on portion of income above a certain income level
- Follows progressive taxation
  - The more income you earn, the higher the % tax rate will be
- Possible solution to reduce income inequality?
- Formula using 2019 Joint Income Taxes
  - Base + Tax Percentage \* (Total Income - Maximum of Lower Bracket)
- Maximum of Bracket
- Base
- Tax Percentage

<details><summary>Speaker notes</summary>

In general, you will pay a lower tax rate for your first X dollars, and a higher rate for dollars earned over X.

</details>

## Slide 50: Top Marginal Tax Rates and Income Inequality

- Is there a relationship?

## Slide 51: US Income Inequality Trends

- Top 1% Share
  - Decreases from 1910 - 1970s
  - Increases from 1970s - present
- Top 10% Share
  - Increases from 1910 - 1930
  - Decreases from 1930s - 1970s
  - Increases from 1970s - present
- Bottom 50% Share
  - Starts at 1960s
  - Increases a bit
  - Decreases from 1970s - present

## Slide 52: US Policies 1910s - 1970s

- Income Inequality Decreasing
- 1910s policies
  - Top 1% started contributing to taxes to make up for reduced revenue from high tariffs
  - Top marginal rate started to increase from 7% to 73%
- Great Depression
  - Top marginal tax rates increased to 94% by 1944
- Great Compression (40s to 70s)
  - Before the Great Depression, the top 1% had ~20% of total income
  - During the Great Depression, the top 1% had ~15% of total income
  - For 10 years after, the top 1% had below 10% of total income
  - For 30 years after, the top 1% had ~8% of total income

## Slide 53: US Policies 1970s - Present

- Income inequality Increasing
- 1970s - High unemployment and inflation, low growth
- Government took action
  - Reduced marginal tax rates (from 70% to 38.5%)
  - Deregulated corporate institutions
  - Unfriendly to labor union memberships (membership decreased by half within 30 years)
- The share for the bottom 50% decreases and share for the top 1% increases
- Financialization

<details><summary>Speaker notes</summary>

Economic Recovery Tax Act of 1981 by Reagan

</details>

## Slide 54: Summary of Income Inequality

- Between 1910s and 1970s, income inequality decreased. Between 1970s and the present, income inequality is increasing
- There is an inverse correlation between top marginal tax rates and income inequality
- Other key factors of income inequality are unemployment rate and population growth
- Income inequality is increasing around the world

<details><summary>Speaker notes</summary>

E

</details>

## Slide 55: Compare US states?

- https://wid.world/country/california/       https://wid.world/country/new-york/

## Slide 56: Realtime Inequality Demo

- Realtime Inequality
- Methodology

<details><summary>Speaker notes</summary>

P

</details>

## Slide 57: Github!

## Slide 58: Realtime Inequality

- Income Growth in the US (1976-present)
- Wealth Growth in the US (1976-present)
- Disposable Income Growth (COVID-19)
- Real Factor Income Growth (COVID-19)
- How can data be used to motivate
- Public Policy?

<details><summary>Speaker notes</summary>

P

</details>

## Slide 59: Tax Justice Now Demo

- Tax Justice Now

## Slide 60: Tax Justice Demo

- Do one of the following:
- Make your own tax plan
- or
- Compare proposed tax plans
- What do you think has the greatest impact creating a truly progressive tax system? What is the effect of a wealth tax? What’s the impact on tax evasion? Any other thoughts?

## Slide 61: Income Inequality Internationally

Lecture Notebook 7.2

## Slide 62: Unemployment Rates and Inequality

- Positive correlation between income inequality and unemployment rates
- Certain groups are vulnerable to unemployment in countries with high income inequality
  - young workers
  - low-skilled workers
  - workers who had been out of work for a long time
  - Why?
    - Employers prefer skilled workers with relevant experience
- “Unemployment accounts for a large part of the increase in earnings inequality that this country experienced between 1991 and 1998”
- Martín González (Universidad Torcuato Di Tella) and Alicia Menendez (Princeton University)
- Simulate wages across employed and unemployed over time

<details><summary>Speaker notes</summary>

E

</details>

## Slide 63: Unemployment Rates

- Countries with notable unemployment rates
- Brazil (53.9)
- Europe
- US
- Russia

<details><summary>Speaker notes</summary>

E

</details>

## Slide 64: Population Growth and Inequality Discussion

- How does population growth affect economic inequality?
- Growth Phase ( 20th Century)
- Strong correlation between a country’s population growth (measured by birth rates) and its income inequality
  - University of Toronto’s Marijn Bolhuis and University of Oxford’s Alexandra de Pleijt
- Wealth Phase ( 21st Century)
- Finds that low birth rates, rather than high birth rates, are causing today’s income inequality.
  - With lower birth rates, fewer children per couple are being borne, so these children get more of their parents’ inheritance.
  - Thomas Piketty (Economics Professor at  London School of Economics)

<details><summary>Speaker notes</summary>

E

</details>

## Slide 65: Population Growth

- Most countries had a high population growth between 1-2% during the 1980s
- Effects can be seen in the 90s

<details><summary>Speaker notes</summary>

E

</details>

## Slide 66: World Inequality Database Demo

- Data from the World Inequality Database (co-directed by Berkeley Economics Professors Emmanuel Saez and Gabriel Zucman)
- Data collection methodology
  - Fiscal
  - Survey
  - National accounts
- Compared to
  - Self-reported survey data
- Why is relying on just self-reported survey data unreliable?

<details><summary>Speaker notes</summary>

P

</details>

## Slide 67: World Inequality Database Demo

- Data from the World Inequality Database (co-directed by Berkeley Economics Professors Emmanuel Saez and Gabriel Zucman)
- Data collection methodology
  - Fiscal
  - Survey
  - National accounts
- Compared to
  - Self-reported survey data
- Why is relying on just self-reported survey data unreliable?
  - Non-response bias from the top income bracket
  - Limited time span

## Slide 68: Your turn! Go to one of the following:

- WID - World Inequality Database (Income Share of top 10%)
- WID - World Inequality Database (Wealth Share of top 10%)
- WID - World Inequality Database (Carbon Emissions Share of top 10%)
- WID - World Inequality Database (Female Labor Income Share)

## Slide 69: International Income Inequality Data

- Top 10% income bracket

## Slide 70: International Income Inequality Graphed

## Slide 71: International Income Inequality Trends

- Global inequality is rising worldwide
- What explains Russia’s trend?

## Slide 72: International Income Inequality Trends

- Global inequality is rising worldwide
- What explains Russia’s trend?
  - Income inequality increased around 1991
  - Fall of USSR
  - Former USSR officials acquired state property
  - Increase in Russian oligarchs with mass privatization

## Slide 73: US versus Europe Income Inequality

## Slide 74: Income Gini for China

- https://www.statista.com/statistics/250400/inequality-of-income-distribution-in-china-based-on-the-gini-index

## Slide 75: Income Gini for Mexico by State

- https://www.statista.com/statistics/1040573/income-distribution-gini-coefficient-mexico-state/

## Slide 76: Wealth Inequality in OECD

- https://www.oecd-ilibrary.org/docserver/7e1bf673-en.pdf

## Slide 77: Median Income vs GDP per Household

## Slide 78: Compensation / GDP

