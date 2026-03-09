---
title: "Lecture 11 - Econometrics"
type: slides
week: 11
source_path: "/Users/ericvandusen/Documents/Data88E-ForTraining/F24LS/Lecture 11 - Econometrics.pptx"
---

## Slide 1: Lecture 11: Econometrics

- Lecture 11: Econometrics
- Data 88E: Economic Models

## Slide 2: Announcements

- Project 4 will be released - Econometrics
- No Lab

## Slide 3: Academic Dishonesty and Attendance

- Using ChatGPT to paste in code is dishonesty

## Slide 4: Follow-ups

- Last time I ranted about Facebook 2016 and 2020 - its in court today!!!
- Did Facebook lie to shareholders about knowledge of risks?
- Brent Kavanaugh’s best friend Joel Kaplan is Facebook Policy head?

## Slide 5: Fed Meeting tomorrow!

- What is forecast for tomorrow?
- How might the election affect the Fed decisions

## Slide 6: (untitled)

- https://www.spglobal.com/marketintelligence/en/news-insights/latest-news-headlines/fed-s-autonomy-could-be-at-risk-if-trump-wins-in-november-85804211

## Slide 7: Project 2025 has a chapter on the Fed - Chapter 24!

- Do away with Dual Mandate
- https://static.project2025.org/2025\_MandateForLeadership\_CHAPTER-24.pdf

## Slide 8: Today’s class

- Intro to econometrics
- Intro to regression
- Simple linear regression
- Multiple linear regression
- Dummy variables
- Reading econometrics tables
- Project demo
- Upper division econometrics classes

## Slide 9: Memes of the day

## Slide 10: Intro to econometrics

- An important part of economics is understanding the relationship between variables, especially cause-and-effect relationships
  - How does price affect the quantity demanded?
  - How does income tax affect consumption?
  - How does schooling affect income?
- Ideally, we would do randomized controlled trials to see if one variable (independent variable) causes another (dependent variable): gold standard for establishing causality
- But we can’t usually conduct experiments in economics, the best we have is observational data
  - What do you think are some challenges with conducting experiments in economics?
- This is why we use econometrics: we use statistical techniques (like regression) to model relationships between economic variables

## Slide 11: Intro to regression

- Really important tool in econometrics
- Used for modelling relationships between variables
  - E.g. relationship between price and quantity demanded
- 3 main purposes of regression:
  - Describing associations: What kind of association do x and y have? Positive/negative? Strong/weak?
  - Prediction: E.g. by how much will quantity demanded decrease if price increases by $1?
  - Causal inference: Does x cause y or is it just a correlation?

## Slide 12: Simple linear regression

- You hypothesize that there’s some association between two variables – say price and quantity demanded
- Start by creating a scatter plot to visualize the association
  - This gives you an intuition for the association
  - Gives you a sense for whether the association is positive/negative, strong/weak/none
  - There is some randomness: e.g. in the example, y doesn’t always increase as x increases (otherwise the points would all lie on a straight line)

## Slide 13: Simple linear regression

- You can quantify this association by calculating the correlation coefficient or r (DATA 8 review!)
- Review of correlation coefficient:
  - Measure of linear association (won’t work for nonlinear association), i.e. slope is constant with respect to x
  - Value between -1 and 1
  - The sign tells you the direction of the association: positive or negative
  - Take absolute value of r to determine strength of association
  - 0 means no correlation
  - Correlation is not causation!

## Slide 14: Simple linear regression

- How to calculate the correlation coefficient:
  - Step 1: Convert data to standard units
    - Express each value in terms of the number of standard deviations it is from the mean (“distance” from the mean)
  - Step 2: Take the mean of product of x and y

## Slide 15: Simple linear regression

- We can draw a line of best fit through the data: the line that most accurately models the relationship between x and y
- Recall that this line has the form y = slope \* x + intercept
- We can use the correlation coefficient to find the slope and intercept
  - Slope:
  - Intercept:
  - Equation of the line of best fit:                                 → Regression equation
- A note notation: the hats indicate that the values are estimates (rather than actual values)
- (fact: because average of x and y are always on the regression line)

## Slide 16: Simple linear regression

- Can plot the regression line (the line of best fit) using the equation we found
- This is the same as the line of best fit you get by using a function you already looked at: np.polyfit
- Not all the points lie on the regression line because of
  - Random variation
  - The model is not a perfect fit (not perfectly accurate)
- Note: assuming all assumptions of doing linear regression are satisfied

## Slide 17: Avocado Demand in Week 2?   np.polyfit(X,Y,1)

## Slide 18: polyfit

## Slide 19: Simple linear regression

- Use the regression equation to generate predictions for y
- Once we’ve generated predictions, we would be interested in knowing how accurate they are
- We can calculate root mean squared error (RMSE) to determine accuracy:
  - Calculate residuals → Square them → Take the average → Take the square root

## Slide 20: Simple linear regression

- RMSE: on average, how far are your predictions from the actual values?
  - Think about whether your value for RMSE is a large number
  - Most useful for comparing across models
- Regression minimizes the RMSE of your data: of all the lines you can draw through your points, the regression line has the lowest RMSE
  - This is why it’s called least squares (OLS) regression
  - We can use the minimize() function to see this
- Numpy minimizes RMSE when it’s calculating the slope and intercept (e.g. when you use np.polyfit)

## Slide 21: Simple linear regression

- So far: DATA 8 version of regression
- We use the statsmodels library to do regression in Python
  - First, import statsmodels.api as sm
  - Code for regression:
  - To get the coefficients (intercept, slope): result.params

## Slide 22: (untitled)

## Slide 23: Simple linear regression

- Always include intercept term in your model (that is, don’t forget sm.add\_constant)
  - Regression line passes through point of averages (property of the regression line)
  - Best prediction for the average of x is the average of y
  - Including the intercept term makes sure that your regression line passes through the point of averages
  - Also, can’t generally assume that your model doesn’t have a y-intercept

## Slide 24: Simple linear regression

- Important things to focus on in the regression output:
  - Slope (𝛃) : By how much does y increase/decrease when you increase x by 1 unit?
  - Intercept (𝛂) : What is the value of y when x = 0?
    - Not always meaningful (e.g. what are earnings when height is 0 inches?)
  - Confidence interval: Is there a statistically significant association between x and y? (H0: 𝛃 = 0, H1: 𝛃 ≠ 0)

## Slide 25: Simple linear regression

- Confidence interval on coefficient estimates:
  - Related to hypothesis testing:
    - Null hypothesis: No association between x and y  (𝛃=0)
    - Alternative hypothesis: There is an association between x and y (𝛃≠0)
  - Regression output gives you 95% CI → test hypotheses at the 5% significance level
    - If CI contains 0: evidence for the null
    - If CI doesn’t contain 0: evidence against the null
  - Can simulate using bootstrapping
    - Center of the CI: regression slope

## Slide 26: Regression Output ( Statsmodels)

## Slide 27: Regression Output ( SM vs R vs Stata)

- Python (statsmodels)
- R - LM
- Stata

## Slide 28: Regression Output ( Statsmodels)

- This is the amount of variation explained by the model
- Number of Observations - reality check on size of data  N matters!
- coef = 𝛃 = magnitude of estimated coefficient
- std err = variability of estimate of coefficient
- t=  a t-test testing whether 𝛃 = 0
- P>[t]=  probability of that t-test

## Slide 29: Jump to Notebook 1

## Slide 30: Interactive Demo

## Slide 31: (untitled)

## Slide 32: (untitled)

## Slide 33: Multiple linear regression

- These variables that affect the dependent variable and are correlated with the independent variable that are not included in the model are called omitted variables
  - E.g. your socioeconomic background affects your earnings and is positively correlated with how much education you get
- They cause omitted variable bias: cause your estimate of the regression slope to be different from the actual slope
  - What would a positive vs. negative value for omitted variable bias mean?
- Can use multiple linear regression to account for omitted variables (really important purpose of regression!)

## Slide 34: Multiple linear regression

- These variables that affect the dependent variable and are correlated with the independent variable that are not included in the model are called omitted variables
  - E.g. your socioeconomic background affects your earnings and is positively correlated with how much education you get
- They cause omitted variable bias: cause your estimate of the regression slope to be different from the actual slope
- Can use multiple linear regression to account for omitted variables

## Slide 35: Multiple linear regression

- Using statsmodels: just select multiple columns when defining the x-variable
  - Multiple regression slopes: each independent variable has a slope
- Compare slope on independent variable of interest from simple and multiple linear regression to see if it’s overstated or understated
- Example of a MLR model:

## Slide 36: Multiple linear regression

- Mathematical intuition:
- Each slope is the partial effect of the corresponding independent variable on y: effect holding all other independent variables constant

## Slide 37: Multiple linear regression

- Graphically with 2 independent variables: 3D plot with a regression plane (not line)
- Y
- X1 - continuous variable
- X2 - dummy variable
- Can only be 0,1

## Slide 38: Dummy variables

- Dummy variables are variables that take on a value of either 0 or 1 to indicate the presence or absence in a category
  - E.g. smoker or non-smoker, went to college or didn’t go to college
- Also known as indicator variables
- Mutually exclusive: you can only be in one of the categories (you either went to college or you didn’t – not both)
- Collectively exhaustive: you must be in one of the categories (only 2 possible scenarios)

## Slide 39: Dummy variables

- Example of regression model with dummy variable:
  - col is a dummy variable indicating whether or not the person went to college (0 = didn’t go to college)
  - Coefficient on col → difference of mean earnings when col = 1 and col = 0

## Slide 40: Dummy variables

- Beware of the dummy variable trap
- Say you include all possible variables for a dummy variable
  - E.g. col (for whether or not you went to college) and notcol (for whether or not you didn’t go to college)
  - Example:

## Slide 41: Dummy variables

- There are infinite solutions for the coefficients (slopes and intercepts)
- This is because the independent variable in the model have a perfect correlation – perfect multicollinearity
- Happens any time one independent variable is a linear combination of another
  - E.g. if you have age and schooling in your model and age = schooling + 3
- Use pd.get\_dummies() to convert categorical variables to dummy variables

## Slide 42: Jump to Lec NB2

## Slide 43: Reading econometrics tables

- In papers, the results of econometric analysis are summarized in regression tables
- Example of regression table from project 4:

## Slide 44: Reading econometrics tables

- In papers, the results of econometric analysis are summarized in regression tables
- Example of regression table from project 4:
- Filtering Variable
- X1
- X2
- X3,X4,X5…
- Y
- n

## Slide 45: Reading econometrics tables

- Always look at how the dependent variable is being measured: in the example, the authors are taking log earnings
  - Recall what you learned about semi-log demand curves: interpreted as % change in y per unit change in x
- Look at what is being controlled for, e.g.
  - They are estimating separate models for men and women so they are controlling for gender
  - In the second column for each group, they are controlling for test scores

## Slide 46: Project demo

- In this project, you will use regression to analyze the relationship between a person’s height and earnings
- Based on a study conducted by Anne Case and Christina Paxson (very interesting study!)
- Divided into 3 parts: simple linear regression, multiple linear regression and reading econometrics tables

## Slide 47: Upper division econometrics classes

- ECON 140: Economic Statistics and Econometrics
  - Edwards - in R in Fall
- ECON 141: Econometric Analysis (with Linear Algebra)
- ECON 142: Applied Econometrics and Public Policy
- ECON 143: Econometrics: Advanced Methods and Applications ( New)
- ECON 144: Financial Econometrics
- STAT 153: Time Series
- ECON 148:  Data Science for Economists
- Learn more here: http://guide.berkeley.edu/courses/econ/

## Slide 48: Aside - Econometrics vs ML

- The idea of Econometrics is that the model has an underlying structure
- Economic Theory gives us a reason to structure the model
- We seek to explain the effect of X on Y
- We also need to hold multiple variables constant
- We will adapt a lot of techniques to get an unbiased estimator
- OLS is the starting point - variations depart from there
- Machine Learning - we don’t need to have an underlying model
- Whatever does the best job in prediction! Or modeling, or classification...
- ML has many models that can inform econometrics!
- Random Forest, Network Graph theory, nonparametric approaches
- Neural Networks

## Slide 49: Terminology - an aside econometrics vs ML

- Dependent Variable ( Y)  ~  predictor and predicted values
- Independent Variable ( X)   ~ regressors, explanatory variables     ~ coefficients, betas
- X  in the ML world might be called “model features”
- Y in the ML world might be called “target”
- 0-1 variables when  they are in the Y variable - call for a different class of models- especially Logit model - in ML this would be a classifier model
- 0-1 dummy variables - in the X variable would be  called “one hot encoding” of categorical variables
- ML - split up your datasetTraining data - used for training the modelTesting data - used for measuring the accuracy of your model

## Slide 50: Matrix notation for Linear Regression

- https://online.stat.psu.edu/stat462/node/132/
- Formula for Data Model
- Formula for OLS

## Slide 51: SKlearn

