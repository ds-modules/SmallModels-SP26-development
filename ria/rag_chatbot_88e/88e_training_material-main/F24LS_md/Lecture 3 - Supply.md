---
title: "Lecture 3 - Supply"
type: slides
week: 3
source_path: "/Users/ericvandusen/Documents/Data88E-ForTraining/F24LS/Lecture 3 - Supply.pptx"
---

## Slide 1: Data 88: Economic Models

- Lecture 3: Supply and Market Equilibrium

## Slide 2: (untitled)

## Slide 3: Announcements

- Today’s lecture:
- Supply - Classic things you have seen in Econ 1 or Econ 100 - done in Python
  - Cost Curves - cubic model - data tables
    - using data tables to make cost curves
  - Electricity Market in California
    - a supply curve example
- Equilibrium
  - Symbolic Python
    - Solve for market equilibrium, 2 equations 2 unknowns, using python
  - A simple data application
    - A table for the demand curve for oranges
- Next time
  - Government / Tax / Welfare

## Slide 4: Office Hours

- Monday 2-4   101B-A Warren
- Tuesday 12-2 101B-A Warren
- Refer to Ed for updated information

## Slide 5: Extra Credit for Answering Ed questions!

- Good Samaritan Policy
- We will be tracking who contributes to answering other people’s questions on Ed
- We will apply this to your Participation Grade !

## Slide 6: Confusion in Lab 1

- Which is the cause, which is the effect?
- Which is the X, which is the Y
- Qd = Q(P)
- Consumers take a price and decide how much to consume
- In Econ - we put Price on the Y axis, but think of it as the “X”
- Consumers / Firms - “take a price” and respond to it by setting quantity

## Slide 7: There’s 6 Lecture Notebooks	!  (Sorry)

- Supply - with Cubic function
- Supply - with fixed table
- California Energy
- CAISO - really hot Tuesday
- Symbolic Python

## Slide 8: Cost and Supply Curves

## Slide 9: Starting off in the Abstract- Cost Curves

- C(Q) - Cost function
- Total Costs of producing a Unit of Output
- Average Costs - C(Q) / Q
- Marginal Costs - Cost to produce an additional unit
  - MC = Δ TVC/ Δ Q
  - Or Equivalently  MC = Δ TC/ Δ Q

## Slide 10: Varian Chapter 21

## Slide 11: https://econvideos.ucsd.edu/video\_handbooks/Intermediate-Microeconomics-Video-Handbook/media/w22g3.html

## Slide 12: Fixed versus Variable - Short Run

- Variable Cost - inputs can be adjusted or changed in short run
  - AVC(Q) = C(Q)/Q
  - Labor, raw materials, etc.
- Fixed Costs - (capital, land) cannot be adjusted in the short run
  - cost of producing nothing
  - Overhead like machines, land, etc.

## Slide 13: Total Costs  - what shape is this and what is the “intuition”

- Quantity of Output
- C(Q)
- $

## Slide 14: Cost Curves

- Total Costs
- Quantity of Output
- C(Q)
- $
- $/q
- $/q
- AC(Q)
- MC(Q)
- Average Costs
- Marginal Costs
- q

## Slide 15: Lecture Notebook - Cubic Function

- Specify parts of cubic function

## Slide 16: Marginal Cost crosses through bottom of Average Cost

- at MC < AC   AC is decreasing
  - The cost of each additional unit is less than the average cost
- at MC> AC    AC is increasing
  - The cost of each additional unit is more than the average cost
- at MC=AC      AC is flat
  - The cost of each additional unit does not change the average

## Slide 17: U Shaped Picture

- AC
- AVC
- MC
- Q
- $ / unit

## Slide 18: Where does Price Come in?

- AC
- AVC
- MC
- Q
- $ / unit
- P
- Q(P)

## Slide 19: Determinants of Supply

- Firm will produce only if Price is above AVC for some quantity
  - If firm decides to produce following the rule above, it will produce till the quantity when Price <= MC

## Slide 20: Jump to Notebook ( & Textbook)

- Market Supply is a horizontal sum of Firm supply
- Build a Cost Table
  - Start with Fixed and Variable Costs for a range of outputs
  - Total Cost
  - Average Variable Costs, Average Fixed Costs, Average Total Costs
  - Marginal Costs   (np.diff)
- Interactive with Production coming from Price

## Slide 21: Python techniques - np.diff

## Slide 22: Python techniques - CSAPS

- CSAPS - Cubic Spline Approximation
- https://csaps.readthedocs.io/
- ( From Matlab from Fortran )

## Slide 23: https://www.aaea.org/UserFiles/file/AETR\_2021\_003RProofFinal.pdf

## Slide 24: Lecture Notebook

## Slide 25: Case Study - Market Supply - California Electricity Strategy Game

## Slide 26: Case Study - 2001 Energy Crisis

## Slide 27: Enron and the California blackout of 2001

- Private energy trading firm
- Hold production off market
- Idle plants
- Reduce transmission

## Slide 28: Energy Strategy Game - How does the game work

- Groups are in teams - that own a portfolio of plants
- Plants have different Variable costs and production capacities
- The day is divided into four time periods
- Teams look at a forecast for each time period and bid how much power they plan to supply
- Market clears at power supplied, and quantity demanded
- Market Price yields profits for each plant, and each portfolio

## Slide 29: Energy Game

- Today we are just looking at the Supply Curve
- Variable Costs per unit of electricity produced
- Start by ranking all producers ( plants ) by their variable  costs from low to high
- Graph with width of bar being quantity supplied in MW
- Profits = Price - VC
- Where price is determined by Market Demand vs Market Supply
- (This Energy grid is 20 years out of date)
  - No solar, no wind, no batteries

## Slide 30: https://www.energymarketgame.org/


Environmental Economics and Policy 147
		Prof Meredith Fowlie

- https://www.energymarketgame.org/
- Environmental Economics and Policy 147
- Prof Meredith Fowlie
- If you are interested after Lecture Notebook:

## Slide 31: Case Study

- Electricity Market for California  -  all of the plants supplying electricity into the grid
- California Independent Systems Operator - a market for electricity
  - ISO controls 300 million MWH  per year
  - ~ 80 percent of California’s Energy
- Is Electricity an Undifferentiated Commodity? Yes but….
  - Time of Day Matters
  - Carbon intensity matters?
  - On Demand vs Renewables

## Slide 32: ISO Control Room in Folsom CA

- http://www.caiso.com/Documents/2014-2016StrategicPlan-ReaderFriendly.pdf

## Slide 33: Aside - Forecasting Peak Demand

- http://www.caiso.com/Documents/2020SummerLoadsandResourcesAssessment.pdf
- http://www.caiso.com/Documents/2022-Summer-Loads-and-Resources-Assessment.pdf

## Slide 34: Simulations of Possible Outcomes

## Slide 35: (untitled)

## Slide 36: Forecasting!

- https://www.caiso.com/about/Pages/Blog/Posts/Reliability-and-the-growing-importance-of-forecasting.aspx

## Slide 37: Simulations of Running out of Power

## Slide 38: CAISO Dashboard - Demand for Electricity

- https://www.caiso.com/TodaysOutlook/Pages/index.html

## Slide 39: Net Demand - the duck curve ?

## Slide 40: April 17 - 2024 - Net Demand is Negative

## Slide 41: Negative Prices

- Only for a few hours of the day?
- How much to subsidize, build batteries?
- How much to pay homeowners for the electricity?
- What about when everyone moves to electric cars?
- “Dumping” electricity on other states?
- “Disconnecting” from the grid for a few hours

## Slide 42: September 6, 2022 - a very hot day

## Slide 43: (untitled)

## Slide 44: Demand Event - send out a text to everyone

## Slide 45: Insert PollEV

- How much do you think is nuclear?
- What do you think happens when there’s too much solar?

## Slide 46: (untitled)

## Slide 47: (untitled)

## Slide 48: (untitled)

## Slide 49: Supply - Right Now and Yesterday

- https://www.caiso.com/TodaysOutlook/Pages/supply.html

## Slide 50: Diablo Canyon - PGE - two 1.1 MW reactors

- Expires November 2024 and August 2025
- But at the last minute…
- Got extended to 2030

## Slide 51: PGE Transmission Lines

- 2023 Schatz Center Wind Transmission Report

## Slide 52: Burea of Ocean Energy Management

- BOEM Presentation

## Slide 53: BOEM

## Slide 54: Wind Lease Areas were auctioned in 2022

## Slide 55: Last year in NYT - who should pay for rooftop solar?

- https://www.nytimes.com/2022/01/24/business/energy-environment/california-rooftop-solar-utilities.html

## Slide 56: Supply - 24hrs - Yesterday

- https://www.caiso.com/TodaysOutlook/Pages/supply.html

## Slide 57: The Grid has Batteries - and Huge Imports

## Slide 58: Batteries vs Gas Plants

- https://www.nytimes.com/2020/09/03/business/energy-environment/california-electricity-blackout-battery.html

## Slide 59: CO2 Dashboard!

## Slide 60: CO2 Trend over time

## Slide 61: Price Map - Needs some explanations

## Slide 62: On to SymPy - Symbolic Python

- Intro and why
- Solving a system of equations
- 2 Equations and 2 Unknowns
- Q as function of Price - Demand
- Q as function of Price -  Supply
- Q\* and P\* are the unique solution

## Slide 63: And On the the Lab

- Oranges Supply and Demand
- Supply - consumption over a period of years
- Demand - they just make it up !

## Slide 64: Pears vs Oranges 1

- https://drive.google.com/file/d/1Exya7KgLoU9wLj4Y6TJNI-PmH5zZVj9w

## Slide 65: Oranges - the data table!

