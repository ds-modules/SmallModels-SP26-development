---
title: "Lecture 5 - Production C-D"
type: slides
week: 5
source_path: "/Users/ericvandusen/Documents/Data88E-ForTraining/F24LS/Lecture 5 - Production C-D.pptx"
---

## Slide 1: Data 88: Economic Models

- Lecture 5: Production & Cobb - Douglas

## Slide 2: Announcements - Project 2

- Congrats on finishing Project 1!
- Project 2 will be released! It is a bit harder than project 1, so be sure to start early.
- Lab 4 will be released, it is due before next class. It has a portion where you upload a dataset on your own - will be useful for data analysis beyond this class!
- Don’t cheat on assignments - this includes copy-pasting things from the textbook. It’s ok to consult the textbook, but write the concepts in your own words please.  Don’t paste in answers from GenAI

## Slide 3: Sympy and fraction

- You can add the following to the sympy commands to have them report numeric / float instead of fractions
- .N()
- .evalf()

## Slide 4: np.arange()

- numpy.arange([start, ]stop, [step, ]dtype=None)
- Parameters:
- start (optional): The starting value of the sequence. The default value is 0 if not provided.
- stop: The end value (exclusive). This means that the generated array will include numbers up to, but not including, the stop value.
- step (optional): The spacing between values. The default step is 1.
- dtype (optional): The data type of the output array. If not specified, it infers the type from the input values.
- Returns:
- A NumPy array of evenly spaced values between start and stop with spacing defined by step.
- arr = np.arange(0, 1, 0.1)
- [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]

## Slide 5: Lecture 5 Outline

- Overview of Production
  - Factors of Production
  - Total Factor Productivity
  - Cobb-Douglas Production Functions (3D SURFACES!!!!!)
  - Cross-Country Analysis ( this is what the project is)
- Derivation
  - Cobb Douglas
  - Log of Equation
  - Solve for one variable
- Intuition
  - Shifts in A (SLIDERS!!!!!!!)
  - What does alpha mean?
  - Returns to scale

<details><summary>Speaker notes</summary>

Evaluation will take a while

</details>

## Slide 6: Mathematical Model

- A mathematical relationship  ~  a representation of a real world phenomenon
- How to describe a system with ( few) variables
- Assumptions - How mathematical properties can drive our model results
- Data pipeline - how to transform our data to fit the model
- https://www.econgraphs.org/graphs/micro/consumer\_theory/preferences\_and\_utility/cobb\_douglas\_utility\_3d

## Slide 7: (untitled)

## Slide 8: Name all of the macroeconomic indicators you know

<details><summary>Speaker notes</summary>

PPI Inflation Unemployment, GDP, Imports Exports Interest Rates, Currency Valuation, Labor Force

</details>

## Slide 9: What is GDP?

- How do we calculate it?

<details><summary>Speaker notes</summary>

Three ways to calculate

</details>

## Slide 10: Production in the Economy

- What is production?
- How do we quantify a country’s production?
- Image source: https://static.seattletimes.com/wp-content/uploads/2017/09/194a9290-98a7-11e7-a238-1aeeb8c083e1-780x405.jpg
- Why would it be important to calculate this?

## Slide 11: (untitled)

## Slide 12: Inputs (or Factors) of Production

- Let’s make a list of all the resources that go into producing and selling an iPhone:
- Chips
- Cobalt
- Lithium
- China - labor assembly
- California - labor design
- intellectual property
- Image source: https://store.storeimages.cdn-apple.com/4982/as-images.apple.com/is/refurb-iphoneX-spacegray\_AV1?wid=1144&hei=1144&fmt=jpeg&qlt=80&op\_usm=0.5,0.5&.v=1548459945536

<details><summary>Speaker notes</summary>

Cobalt
Copper
Education
Worker-hours
Aluminum
Computers
Software
Ships
Trucks
Materials

</details>

## Slide 13: 40 Bn CHIPS ACT Funding  - TMSC Arizona

## Slide 14: Fundamental Idea - simplify for a model

- A nation's output is a function of the amount of the factors of production that are utilized in its economy; that is to say output is a function of labor and capital, scaled by some value A.
- What could A be? Why is it on the outside of the production function?

## Slide 15: Economic Model

- Could this be for a firm?
- Compare across firms?
- Could this be for an entire country ?
- Compare across countries ?
- Could this be for a State?

## Slide 16: Total Factor Productivity (TFP)

- Technology or research and development
- A country with a high TFP can producer far more goods and services than another country with a lower TFP but the same amount of capital and labor
- Differences between TFP and other factors of production
  - “Scales” production by some factor A - creates proportional increase in output
  - Technology is non-rivalrous
  - Technology is non-excludable
  - Ordinal measure

## Slide 17: How the equation was formulated

- Based on the graphs to the right, what would F(K, L) look like?
- Image source: Hawkins’ 100B; Lecture 4, Slide 9

## Slide 18: Cobb Douglas Orginal Data Visualization - 1928

- https://assets.aeaweb.org/asset-server/journals/aer/top20/18.1.139-165.pdf
- Cobb, C. W.; Douglas, P. H. (1928). "A Theory of Production" . American Economic Review. 18(Supplement): 139–165. JSTOR 1811556. Retrieved 26 September 2016.

## Slide 19: Plotting the Model outcome vs Actual outcome

## Slide 20: Analyze the error rate of the model

## Slide 21: Cobb-Douglas Production Functions

- For simplicity sake, Cobb Douglas assume constant returns to scale. That is, alpha + beta = 1. This allows us to rewrite the function in the following way:
- Where else do we see this function?

## Slide 22: Lecture Notebook 1

## Slide 23: Lecture Notebook

## Slide 24: Capital

- A monetary value of the stock or value of all productive, physical assets in an economy (not bonds, stocks or other financial products)
- Increasing returns to capital
- At a decreasing rate
- Fix A, L, alpha
- Vary K

## Slide 25: Behind the textbook

## Slide 26: (untitled)

- Beta = none?
- np.arange(x,y,z)

## Slide 27: MPK

- Diminishing marginal returns to capital
- What is the intuition behind this?

## Slide 28: Labor

- The number of hours worked by all individuals who are willing and able to work within a country for a given year
- Increasing Returns to Scale

## Slide 29: Marginal Product of Labor

- Diminishing Marginal Returns to Labor
- What is the intuition behind this?

## Slide 30: Cross-Country Comparisons

- Cobb and Douglas found that a country’s output can be modeled very well as a weighted average of the log of capital and labor.
- How can we rearrange the function to only be in terms of one variable?

## Slide 31: Cross-Country Comparisons (cont.)

- How can we use the above function to compare countries?

## Slide 32: Shifts in A

- Supply or Technology shocks
  - Natural environment
  - Energy prices
  - Significant financial crises
- What would happen to output?

## Slide 33: What does alpha mean?

- Output elasticities of capital and labor: measure the responsiveness of output to a change in the levels of either labor or capital, holding all else constant.
- Let us assume constant returns to scale. What does alpha > 0.5 mean?

## Slide 34: Returns to scale

- Constant Returns to Scale: alpha + beta = 1
- Decreasing Returns to Scale: alpha + beta < 1
- Increasing Returns to Scale: alpha + beta > 1
- What would the last two look like?

## Slide 35: Notebooks

- Textbook Chapter - in HTML  https://data88e.org/textbook/content/04-production/shifts.html
- Penn World Tables - Cobb Douglas
- Project 2

## Slide 36: Penn World Tables File

- Download from website
- Project - transform to csv to read into datascience tables
- ( Lab - read Excel directly to pandas - then transform to table)
- Upload to website

## Slide 37: Penn World Tables

- https://www.rug.nl/ggdc/productivity/pwt/?lang=en

## Slide 38: Penn World Tables in the Economist

- Another tension in China’s push for self-reliance concerns productivity. Total factor productivity (ie, the amount of output per unit of labour and capital) has barely grown under Mr Xi, a marked deceleration from before the financial crisis (see chart 6). The government believes that aiming for self-sufficiency in high-tech industries will encourage innovation and so boost productivity. In fact, the opposite is more likely. In its efforts to boost domestic champions and spur trade with friendly countries, the government will probably end up conferring advantages on firms that are not the most efficient or capable suppliers of a given product, thereby denting productivity. Because lifting productivity is the only lasting way to raise living standards, that is a worrying prospect.

## Slide 39: Penn World Tables

- Making prices comparable across economies - USD
- Different measures of GDP

## Slide 40: Go to textbook page

## Slide 41: Fed Meeting !!  Last Wednesday

## Slide 42: They release a set of forecasts

- https://www.federalreserve.gov/monetarypolicy/files/fomcprojtabl20240918.pdf

## Slide 43: Fed Dot Plot

- Each Dot is an estimate
- With a lot of data and modeling

## Slide 44: Predictions

- GDP
- Unemployment
- Inflation

## Slide 45: Expected Inflation

## Slide 46: What does Powell / Fed care about ?

- Inflation
- CPI
- PCE
- Core PCE
- Unemployment
- Seasonally adjusted
- Rates
- https://fred.stlouisfed.org/series/FEDFUNDS

## Slide 47: So many data series

## Slide 48: PCE - Release Data is Friday

