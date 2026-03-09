---
title: "Lecture 6 - Utility and Latex"
type: slides
week: 6
source_path: "/Users/ericvandusen/Documents/Data88E-ForTraining/F24LS/Lecture 6 - Utility and Latex.pptx"
---

## Slide 1: Data 88E: Economic Models

- Lecture 6: Utility (ft. LaTeX)

## Slide 2: Announcements

- Lab 5 will be posted ; it will be due this coming Tuesday PM
  - Utility Optimization from Today
  - LaTeX in Jupyter Notebook
- Remember project 2 !
- There’s optional Zoom office hours on Friday!
- Lecture Videos - sorry - I’ll just post to playlist

## Slide 3: Change the Path

- Update the path for this semester
- Go and find the folder in /tree
- In Jupyter ! gets you to command line

## Slide 4: (untitled)

## Slide 5: Lecture 5 Outline - Consumer Choice

- What is Utility?
- Properties of Utility
- Utility Functions
- Consumption Bundles
- Indifference Curves
- Budget Constraints
- Constrained Utility Maximization
- Demand Curves
- QuantEcon Notebook
- What is Latex? What is Markdown?

## Slide 6: What is Utility?

- Abstract and hard to measure
- “Measure of Satisfaction, Worth, Value”
- Units to measure utility??
- Utility is a method to model how individuals make decisions
- A starting point for math to model economic choices

## Slide 7: Properties of Utility

- Ordinal: ordering matters, but the numerical value in and of itself does not
  - E.g. we prefer something with 3 utility units over 2 utility units, but 3 utility units in and of itself does not have any meaning!
  - Utility only has meaning relative to other goods
- “Consumers are rational”
  - They try to maximize their utility
- Our utility is dependent on the goods we consume, and they may affect each other
  - Consumption of left and right shoes
  - Consumption of doughnuts and croissants
- Transitivity: If A is preferred to B, and B is preferred to C, then A is preferred to C.

## Slide 8: Properties of Utility

- Non-negative marginal utility: More is better
- Decreasing marginal utility

## Slide 9: Calculus

- First Derivative is positive - increasing utility
- Second Derivative is negative  - at a decreasing rate
- Math Models - what functions work ?

## Slide 10: Utility with 1 Good

- X: quantity of goods consumed
- Y: utility
- Does this satisfy our 2 properties?
- Non-negative marginal utility
- Decreasing marginal utility
- Increasing at a decreasing rate

## Slide 11: Utility Functions

- Utility functions consider many goods at a time:
- Ideally, when modeling a consumer’s utility we would like to consider all the goods in the world, since relationships may exist between any combination of them. But this is infeasible :(
- Instead, we simplify the world to assume that consumers only seek to maximize their utility between 2 goods.
- A ‘basket’ of goods that the consumer chooses to purchase
- We will simplify our choice set into quantities of each of 2 goods

## Slide 12: Cobb Douglas Utility Function

- Does it satisfy our 2 properties?
- Non-negative marginal utility
- Decreasing marginal utility for both good 1 (x1) and good 2 (x2)

## Slide 13: Consumption Bundles

- A ‘basket’ of goods that the consumer chooses to purchase
- We will vastly simplify our choice set into only 2 goods
- X\_1 = 2
- X\_2 = 3

## Slide 14: Each point represents a consumption bundle

- x\_1=10, x\_2=15
- Coffee
- Doughnuts

## Slide 15: Indifference Curves

- Each axis represents the quantity of a good we consume
- 2 axis = 2 goods
- Indifference curve: a line on the graph that yields the same utility.

## Slide 16: Different Indifference Curves

- A
- B
- C
- Which consumption bundle do consumers prefer the most?

## Slide 17: Different Indifference Curves

- A
- B
- C
- Bundle B is preferred to A (B offers a higher utility than A)
- Bundle C is indifferent to B (they yield the same utility)
- Thus, Bundle C is preferred to A

## Slide 18: Different Indifference Curves - Outwards is better

- A
- B
- C

## Slide 19: Indifference Curves Cannot Cross

- A
- B
- C
- D

## Slide 20: PollEV

## Slide 21: Indifference Curves Cannot Cross

- A
- B
- C
- D is preferred to A
- B is thus preferred to A
- C is preferred to B,
- but A is indifferent to C
- => This violates transitivity!
- D

## Slide 22: Indifference Curves are like Contour Maps

## Slide 23: Budget Constraints

- If consumers had no budget constraints, then they would always pick the infinitely rightmost curve.
- Income
- Price of good 1
- Quantity of good 1
- Price of good 2
- Quantity of good 2
- We will assume that consumers spend every cent they have to maximize utility.

## Slide 24: Graphing the Budget Constraint

- Donuts
- Coffee
- (M/p\_1,0)
- (0, M/p\_2 )
- Spend all your money on coffee
- What we can afford
- Spend all your money on donuts

## Slide 25: Graphing the Budget Constraint

- Donuts
- Coffee
- Less income
- More income

## Slide 26: Properties of Budget Constraints

- For any given income level, the possible points to satisfy the budget constraint forms a line.
- Slope: negative ratio of 2 prices (write x\_2 in terms of x\_1)
- Intercepts: when only 1 good is consumed.

## Slide 27: An Example

- M = $5
- P\_coffee = $2    All coffee = 2.5
- P\_donuts = $1     All donuts = 5
- Donuts
- Coffee
- (2, 1.5) also on budget constraint: 2\*1 + 1.5\*2 = 5

## Slide 28: What if...

- Income doubles (M = $10)?
- Price of coffee increases to $2.5?
- Price of donuts increases to $2?
- How does the slope and intercepts change in each of these scenarios?

## Slide 29: 1)  Income Doubles (Y=10)

- 1)  Income Doubles (Y=10)
- 2)  Price of Coffee increases (Pc=2.5)
- 3)  Price of Donuts increases(Pd=2)
- Donuts
- Coffee

## Slide 30: Constrained Utility Maximization

- Consumers will choose the bundle that gives them the highest utility that they can afford.

## Slide 31: Which Bundle Will Consumers Pick?

- B
- A
- C

## Slide 32: Which Bundle Will Consumers Pick?

- B
- A
- C
- Bundles
- A and B - can afford
- C - cannot afford
- Utilities
- C > B > A

## Slide 33: Which Bundle Will Consumers Pick?

- B
- A
- C
- Consumer will choose where the indifference curve is tangent to the budget constraint
- No point on the higher curve is affordable
- All points on lower curve are inferior

## Slide 34: Demand Curves

- Change in price of a good: doughnuts
- Hold price of Good 2 constant (coffee)
- Hold M constant (income)
- Vary price of Good 1 (doughnuts)

## Slide 35: Red = price of doughnuts = 2$
Blue = price of doughnuts= 1.5$
Green = price of doughnuts = 1$

- Red = price of doughnuts = 2$
- Blue = price of doughnuts= 1.5$
- Green = price of doughnuts = 1$
- Donuts
- Coffee

<details><summary>Speaker notes</summary>

If Y is 5$ and Price of coffee is 2$/cup  - you will always be able to buy 2.5 cups of coffee
Red = price of good 2 = 2$
Blue = price of good 2= 1.5$
Green = price of good 2= 1$

</details>

## Slide 36: (untitled)

<details><summary>Speaker notes</summary>

For each of these budget lines -
we can find the point of tangency 
and see how much is consumed - find the quantity that corresponds to each price

</details>

## Slide 37: A Demand Curve is Born

## Slide 38: Demand Shocks and Shifts

- Hold prices of both goods to be constant
- Income increases: budget constraint shifts, allowing us to reach a higher level of utility

## Slide 39: Y= Income increases

- Y= Income increases

<details><summary>Speaker notes</summary>

Lets look at a change in Income
The budget line shifts out in a parallel slope
We get to a higher indifference curve  ( more money we are happier)
So the quantity of good 2 demanded increases from 1.2 to 1.5

</details>

## Slide 40: A New Demand Curve is Born

## Slide 41: Scooby meme

## Slide 42: Visualizing Cobb-Douglas

- Check out these 3-D
- https://data-88e.github.io/textbook/content/05-utility/utility.html
- https://data-88e.github.io/textbook/content/05-utility/budget-constraints.html

## Slide 43: For the Advanced

- Lagrangian

## Slide 44: Optimization

- We are working on finding a local maximum for 2 goods
- In a well defined space with mathematical properties
  - ( Cobb Douglas)
- What about for n - goods?
- What about for non - convex functions?
- A separating hyperplane?
- What does the lambda mean?

## Slide 45: Demo - Notebook for Consumer Choice - quantecon.org

## Slide 46: QuantEcon Data Science in Python

## Slide 47: Optimization - Scipy -

## Slide 48: (untitled)

## Slide 49: (untitled)

## Slide 50: (untitled)

## Slide 51: Why LaTeX?

- We want to be able to write math equations
- We want to format documents beautifully and ‘academically’
- Open source, shareable, and cross platform software
- We think it’s a good thing for you to know about
- Economists use it
- Economics grad students use it
- Upper Div CS classes use it

## Slide 52: There are easier approaches - hostmath.com

## Slide 53: Let’s go make an overleaf account
https://www.overleaf.com/

“Sign in with Google”
Use UC Berkeley Account

## Slide 54: Overleaf - start a new file

- Copy and paste text into new document on Overleaf
- Or… download file and upload to Overleaf
- You can upload a template
- You can import other folks files
- You can do all of the formatting of the file
- Sections, Headers, Tables
- Journal Article templates
- Quiz and Homework templates

## Slide 55: Good things to know about LaTeX

- Language uses \ a lot to denote actions or unique characters
- Everything that’s part of a symbol or function is wrapped within {}
- E.g. \frac{5}{8} == ⅝
- E.g. \section{This is a section header}

## Slide 56: Titles and Imports

- TLDR: don’t worry about these

## Slide 57: Sections and Subsections

- \section{<SECTION NAME>}
- Without numbers: \section\*{<SECTION NAME>}

## Slide 58: Making a List

- \begin{enumerate}
- \item …
- \item …
- \end{enumerate}

## Slide 59: Adding Images

- Put this in the title/imports:
- \usepackage{graphicx}
- \graphicspath{ {./images/} }
- To add an image:
- Make sure an images folder is created, then to include image.png:
- \includegraphics{image.png}

## Slide 60: Typesetting Equations

- In-line: $<LATEX FORMULA>$
- As its own block: there are a few ways to do this. Two of my most commonly used:
- $$<LATEX FORMULA>$$
- \begin{align\*}<LHS> &<OP> <RHS> \\etc.
- \end{align\*}

## Slide 61: Formulas

- Let’s do four examples:
- The quadratic formula
- Partial Derivative:
- Integral: 			(Does this seem familiar?)
- Simplify 		 	   in 2 steps using align

## Slide 62: You can paste this into Jupyter

- For standalone formulas:
- $$YOUR EQUATION HERE$$
- For in-line typesetting:
- $YOUR MATH HERE$

## Slide 63: And Markdown ( bc Juptyer)

- https://www.markdownguide.org/cheat-sheet/

## Slide 64: Markdown - Headings

## Slide 65: Markdown - Style

## Slide 66: Markdown Tables

## Slide 67: (untitled)

## Slide 68: Edgeworth Demo

- Two consumers
- Fixed endowment of two goods
- Indifference curves starting from different corners

