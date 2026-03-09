---
title: "Lecture 12 - Finance"
type: slides
week: 12
source_path: "/Users/ericvandusen/Documents/Data88E-ForTraining/F24LS/Lecture 12 - Finance.pptx"
---

## Slide 1: Lecture 12: Finance

- Lecture 12: Finance
- Data 88E: Economic Models

## Slide 2: Announcements

- Lab will come out later today
- Lab next week for environmental

## Slide 3: Today’s class

- Back to Facebook Transparency  ( more ranting)
- Personal Finance and Data Science
  - Evolution of Data Intensive Apps
- Time Value of Money   - Interest rates  - NB
- Options - puts and calls
- Stocks Indices  & Returns - NB
  - Pulling in data from an API  - Yahoo finance
  - Using Pandas to graph

## Slide 4: Fake pro-Harris Group runs false ads in Swing States

## Slide 5: Meta Ad Library

## Slide 6: Ad testing phase

## Slide 7: The ads link to a fake website with a fake agenda

## Slide 8: (untitled)

- Fake Pro-Harris group - Youtube ads to Muslims

## Slide 9: Meme for the day - Nov 22,2023 - Powerball was at 1.9bn

- Expected Value!  EV = X \* P(x)
- Cost of Lottery Ticket =2$
- Odds of winning are 1 in 292.200,000
- Net lump sum payout net of taxes = 706,000,000
- EV = $706m/292m =  $2.41 > $2 cost
- Potentially for the first time ever …

## Slide 10: Financial Economics and Data

- Finance - markets where money is on both sides of the exchange
- Decision theory applied to investment decisions
- Allocation across space and across time
  - Asset Pricing  - Investment decisions
  - Corporate Finance  - allocations of capital, structures of corporations
- “Present Value” - allocation over time periods
- “Expected Value” - allocation across uncertain outcomes

## Slide 11: Personal Finance and Data Science

- Saving and Investing - more options available to more people than ever before
- Rapid evolution & innovation of how consumers relate to markets
- On their mobile devices and appears to be free!
- How do we actually pay for these innovations?

## Slide 12: Cash Apps

- https://www.businessofapps.com/data/venmo-statistics/

## Slide 13: How does Venmo make money?

- 3% Merchant Fee on transactions - just like credit cards
- But….
- https://www.businessofapps.com/data/venmo-statistics/

## Slide 14: They do hold  35bn in Customer Funds!

- Venmo & Paypal balances
- “Funds Receivable &  Customer Accounts”
- https://s201.q4cdn.com/231198771/files/doc\_financials/2022/q3/PYPL-Q3-22-Earnings-Release.pdf

## Slide 15: Robinhood - growth of micro-investing

- https://www.businessofapps.com/data/robinhood-statistics/

## Slide 16: Robinhood in Mid- April

- https://www.divergentplanning.com/blog/the-tax-consequences-of-your-kids-trading-on-robinhood

## Slide 17: How does Robinhood make money?

- Payments  for Order Flow - PFOF
- Retail brokerage gets a payment / commission for directing orders to market makers
- Market Makers and Dark Pools
- Huge unregulated pools of trades, within hedge funds, off markets
- Aggregated across users
- Spread between buy and sell
- Can be fractions of a cent on huge volumes
- Retail investor sees free trading,
- But doesnt know how close prices is to trading price

## Slide 18: Dark Pools and Algorithmic Trading

- Huge Market Making pools of trades
- High Frequency trading   - front running markets by thousandths of a second
- Algo Wars - planting hundreds of fake trades to conceal the trades that you are actually going to make

## Slide 19: Algorithmic Trading - is there a retail Python via API version?

- Alpaca API - make apps!
- Algo trade from Jupyter

## Slide 20: Goldfish on Alpaca beats wallstreetbets

- Python trading on Alpaca
- Sentiment analysis on Wallstreetbets
- Train the data via a website
- Michael Reeves 10m views

## Slide 21: Financial Economics

- Share prices, interest rates, exchange rates
- “ where money is on both sides of a trade”  - instead of goods and services
- Decision Making under uncertainty!  - Expected Present Value
- Time
- Risk
- Value of a strategy in comparison to a risk free return
- Build models to value financial instruments  - Mathematical Models
- Validate models  using  data - Financial Econometrics
- Build trading models based on model  - Quantitative Trading

## Slide 22: How do we think about interest rates?

- Government Finance - a macroeconomic variable that can be used for policy
- Government setting interest rates to stimulate/slow economic growth
- But - interest rates are also set in the market place - and reflect decisions
- Time preference - how much would you be indifferent to having $ now vs money in the future  - discounting the future
- 5 % interest - $100 today would be worth $105 in  one year
- We can think about this in terms of a Loan  - cost to borrow
- Savings - payment for savings
- But it can be embedded in a whole range of financial arrangements

## Slide 23: Money has time value

- Would you rather have $100 now or later?
- Interest Rate, r
  - 5% interest rate => r= 0.05
- A deposit of $100 at bank 1 has a future value of $110.25 in 2 years.

<details><summary>Speaker notes</summary>

Continuous compounding -> the more frequent you compound the greater this interest accumulates

</details>

## Slide 24: Making a formula

- n= Number of times per year
- T = year
- Nt = number of times in a given that interest is paid
- How frequency you compound the greater the interest rate
- ( the faster it grows)

## Slide 25: Time Value of Money

- Future Value
- Present Value
- Discount Factor
- Present Value of an asset is the future value of the asset times a discount factor

## Slide 26: How does this play out as a consumer?

- Mortgage payment n=360 r=.03
- Student Loans n=120 r=.025
- Car Loans n=60  r=.04
- Credit Cards  n=??  r=.16
- A = Amortized Payment
- P=Principal
- r=interest rate as a decimal
- n= number of time periods

## Slide 27: Present Value and the value of a Bond

- F = face value
- iF = contractual interest rate
- C = F \* iF = coupon payment (periodic interest payment)
- N = number of payments
- i = market interest rate, or required yield, or observed / appropriate yield to maturity (see below)
- M = value at maturity, usually equals face value
- P = market price of bond.

## Slide 28: There are many types of bonds

- Treasury Bonds
- Savings Bonds
- Agency Bonds
- Municipal Bonds
- Corporate Bonds

## Slide 29: Jump to Notebook on programming interest rates

## Slide 30: (untitled)

## Slide 31: (untitled)

## Slide 32: (untitled)

## Slide 33: (untitled)

## Slide 34: Personal Finance

## Slide 35: Bonus: Taxes in the U.S.

- Source: https://archive.nytimes.com/www.nytimes.com/imagepages/2012/04/13/opinion/sunday/0415web-leonhardt2.html

## Slide 36: Personal Finance: Disclaimer

- I have no qualifications (ex. CFP, CPA) in finance or personal finance. This information is just based on my experience and knowledge I have picked up formally (ex. through undergraduate education) and informally. This is not financial advice, but instead an overview to get you interested in the subject!

## Slide 37: Personal Finance: Priorities

- Savings rate
- Stock/bond split
- Portfolio allocation
- Fees
- Taxes

## Slide 38: Personal Finance: Savings Rate

- How much you save (and when you start saving) will have the largest impact on your long-term finances

## Slide 39: Personal Finance: Stock/Bond Split

- Stocks: generally riskier, but with higher returns on average
- Bonds: generally safer, but with lower returns on average

## Slide 40: Personal Finance: Stock/Bond Split

## Slide 41: Personal Finance: Stock/Bond Split

## Slide 42: Personal Finance: Portfolio Allocation

- You’ve figured out your stock/bond split, but what stocks should you buy? Financial advisors generally suggest a well-diversified set of stocks, which can be easily achieved through ETFs, index funds, or mutual funds. But which ones?
- Pay attention to:
- Diversification
- Fees
- Active vs. passive management (active management is BAD)
- Do NOT pay attention to:
- Past returns

## Slide 43: Personal Finance: Portfolio Allocation

## Slide 44: Personal Finance: Portfolio Allocation

## Slide 45: Personal Finance: Portfolio Allocation

## Slide 46: Personal Finance: Portfolio Allocation

## Slide 47: Personal Finance: Portfolio Allocation

- Source: https://www.schwab.com/learn/story/schwabs-long-term-capital-market-expectations

## Slide 48: Personal Finance: Portfolio Allocation

## Slide 49: Personal Finance: Want to learn more?

- There’s way more to personal finance:
- Taxes (huge)
  - Retirement accounts: 401k, IRA, roth vs. traditional, etc.
- Credit scores
- Debt
- Renting/leasing vs. buying
- Insurance
- Career choice
- Etc.
- Take UGBA 135—my \#1 class recommendation for UC Berkeley students!
- There is UGBA 235 for master’s students

## Slide 50: Notebook on Personal Finance

- And how to make widgets

## Slide 51: Price of a Stock?

- Its just defined by the last time a stock changed hands
- This could be millions of times per day

## Slide 52: Gamestop story

## Slide 53: Indexes

- An index is composite measure of an index composed of a set of stocks
- Dow Jones Index - a select few representative stocks
  - Calculation was costly
- SP 500 - the 500 biggest stocks
  - weighted average, indexed to particular time period
- Nasdaq - almost all stocks listed

## Slide 54: S&P 500 on Yahoo Finance

## Slide 55: Long and Short

- Finance jargons explained:
- Long ≈ Buy
- Short ≈ Sell
- So,
- Long a stock means buy a stock
- Short a stock means sell a stock

## Slide 56: More about Short

- But shorting a stock doesn’t mean you need to own the stock beforehand…

## Slide 57: More about Short

- But shorting a stock doesn’t mean you need to own the stock beforehand…
- Imagine if you think the stock price is going down tomorrow

## Slide 58: More about Short

- But shorting a stock doesn’t mean you need to own the stock beforehand…
- Imagine if you think the stock price is going down tomorrow
- Now (Stock price = $100): Borrow a share from a broker - sell it  - get $100
- Tomorrow (Stock price = $70):If stock goes down over a year $70 - buy it back for $70
- Give the share back to the broker  Made $100 - $70 = $30
- “Short”  - borrow/sell now ~ later buy/give back
- ( and you have to pay some interest on borrowing that stock for 1 year)
- ( and how much collateral do you have to put down)

## Slide 59: More about Short

- But shorting a stock doesn’t mean you need to own the stock beforehand…
- Imagine if you think the stock price is going down tomorrow
- Now (Stock price = $100): Borrow a share from a broker - sell it  - get $100
- Tomorrow (Stock price = $70):If stock goes down over a year $70 - buy it back for $70
- Give the share back to the broker  Made $100 - $70 = $30
- “Short”  - borrow/sell now ~ later buy/give back
- ( and you have to pay some interest on borrowing that stock for 1 year)
- ( and how much collateral do you have to put down)
- You are shorting $100 of this stock at this moment

## Slide 60: Stocks Options

- Stocks: Ownership in the company
- Price: The price of the stock was last traded for
- Long Position: The party that has agreed to buy the underlying in the future
- Short Position: The party that has agreed to sell the underlying stock in the future

<details><summary>Speaker notes</summary>

Stock Price: the price of the last trade / liquid

</details>

## Slide 61: Futures

- The 125th Big Game is around the corner (Nov 19th)! You want to buy a ticket from a “certified” reseller, but you will only have enough cash to pay for a ticket next week. The reseller told you that the ticket price is very volatile.

## Slide 62: Futures

- How to secure a ticket from the reseller?
- The reseller told you that you can “book” a ticket at $110. So you will have to buy a ticket at $110 on Nov. 15th regardless of the market price at the time.

## Slide 63: Futures

- How to secure a ticket from the reseller?
- The reseller told you that you can “book” a ticket at $110. So you will have to buy a ticket at $110 on Nov. 15th regardless of the market price at the time.
- This is called a future contract.
- Futures are a type of derivative contract agreement to buy or sell a specific commodity asset or security at a set future date for a set price.

## Slide 64: Futures

- Would you enter into this future contract at $110 for a Big Game ticket?

## Slide 65: Futures

- Would you enter into this future contract at $110 for a Big Game ticket?
- The market price on Nov 15th turns out to be $80

## Slide 66: Futures - Payoff Diagram

- For those who entered the future contract:
- You buy the ticket at $110 when the market price is $80
- market price
- profit/loss
- 110
- 140
- 80
- -30
- 30

## Slide 67: Futures - Payoff Diagram

- What if the price on Nov 15th is $140?
- market price
- profit/loss
- 110
- 140
- 80
- -30
- 30

## Slide 68: Futures - Payoff Diagram

- What if the price on Nov 15th is $120?
- market price
- profit/loss
- 110
- 140
- 80
- -30
- 30

## Slide 69: Futures - Payoff Diagram

- So the payoff diagram is the following:
- market price
- profit/loss
- 110
- 140
- 80
- -30
- 30

## Slide 70: Option

- What if we want to make sure that the price doesn’t go more than $110, but also want to have the freedom to buy tickets if the actual market price is lower?
- market price
- profit/loss
- 110
- 140
- 80
- -30
- 30

## Slide 71: Call Options

- Call: Gives the holder the right, but not the obligation to buy the underlying stock in the future at a pre-specified price, and this pre-specified price is called the strike price.
- Strike price = $100

## Slide 72: Call Options

- When you buy a call (you are longing a call) with strike price $100,
- If the stock price is $150, you can exercise your call and buy the stock at $100

## Slide 73: Call Options

- When you buy a call (you are in a long call position) with strike price $100,
- If the stock price is $150, you can exercise your call and buy the stock at $100
- If the stock price is $100, you can buy the stock at $100
- If the stock price is $50, you can buy the stock at $50
- Call option is an option rather than a binding contract!

## Slide 74: Call Options

- When you buy a call (you are in a long call position) with strike price $100,
- If the stock price is $150, you can exercise your call and buy the stock at $100
- If the stock price is $100, you can buy the stock at $100
- If the stock price is $50, you can buy the stock at $50
- Call option is an option rather than a binding contract!
- Quick check: you would buy a call (long call) if you think the stock price is going up or going down?

## Slide 75: Call Options

- When you buy a call (you are in a long call position) with strike price $100,
- If the stock price is $150, you can exercise your call and buy the stock at $100
- If the stock price is $100, you can buy the stock at $100
- If the stock price is $50, you can buy the stock at $50
- Call option is an option rather than a binding contract!
- Sanity check: you would buy a call (long call) if you think the stock price is going up or going down?
- Going up!

## Slide 76: Call Options

- You can think about call options as an “insurance”.
- It insures that you don’t need to pay more than the strike price for a stock.

## Slide 77: Calls

- Insurance for people who are short some stock, but this isn’t the most helpful way to think about them.
- Suppose you want to buy and hold a stock, but you aren’t sure if the value of the stock will go up in a desired time frame. Instead of buying the stock and risking that its value will decrease, you could buy a call that gives you the opportunity to purchase the stock at some price that you want, say $100. That way, if the value of the stock goes above $100, instead of missing out on that increase in value, your call gives you the right to purchase the stock at $100, even though it is actually worth more than $100. If the stock is worth less than $100 after some time, you don’t have to do anything and you didn’t lose the value you would have lost if you had purchased the stock.

## Slide 78: Call Options

- Step 1: Buying and selling a call option
- Long Call
- Short Call
- Pay a premium
- $10
- A call option
- Alice
- Bob

## Slide 79: Call Options

- Step 1: Buying and selling a call option
- Long Call
- Short Call
- Pay a premium
- $10
- A call option
- A call option with strike price $100
- $10 cash
- Alice
- Bob

## Slide 80: Call Options

- Step 2: (Scenario 1) At expiration date, the stock price is $120
- Long Call
- Short Call
- $100
- One share of stock
- A call option with strike price $100
- $10 - $120 + $100
- Exercise the option
- Buy one share of stock at $120 and sell it to the other person at $100
- Alice
- Bob

## Slide 81: Call Options

- Step 2: (Scenario 2) At expiration date, the stock price is $80
- Long Call
- Short Call
- A call option with strike price $100
- $10
- the option expires
- Alice
- Bob

## Slide 82: Put Options

- Put: Gives the holder the right, but not the obligation to sell the underlying stock in the future at a pre-specified price, and this pre-specified price is called the strike price.

<details><summary>Speaker notes</summary>

Diagrams are wrong!

</details>

## Slide 83: Put Options

- You can think about put options as an “insurance”.
- It insures that you won’t sell less than the strike price for a stock.

## Slide 84: Put Options

- You want your stock to be able to increase in value over time, but you also don’t want its value to decrease too much.
- One way to achieve this is to buy “insurance” on your stock. You might want an insurance contract that will cover losses if the value of your stock goes below a certain number.
- A Contract that pays you a dollar for each dollar that your stock does below some specified number.

## Slide 85: Put Options

- Step 1: Buying and selling a put option
- Long Put
- Short Put
- Pay a premium
- $10
- A put option
- Alice
- Bob

## Slide 86: Put Options

- Step 1: Buying and selling a put option
- Long Put
- Short Put
- Pay a premium
- $10
- A put option
- A put option with strike price $100
- $10 cash
- Alice
- Bob

## Slide 87: Put Options

- Step 2: (Scenario 1) At expiration date, the stock price is $80
- Long Put
- Short Put
- $100
- One share of stock
- A put option with strike price $100
- $10 - $100 + $80
- Exercise the option
- Buy one share of stock at $100 from the other person and sell it to the market at $80
- Alice
- Bob

## Slide 88: Put Options

- Step 2: (Scenario 2) At expiration date, the stock price is $120
- Long Put
- Short Put
- A put option with strike price $100
- $10
- the option expires
- Alice
- Bob

## Slide 89: Put example

- So for instance, if you own a stock that trades at $110, and you don’t want to lose more than $10 in value from owning the stock, you might buy a put with a strike of $100.
- The strike of the put is this pre-specified number below which you don’t want to lose money.
- Now, let’s say your stock starts going down in value. Going down from $110 to $100, there’s nothing that your put can do.
- But starting at $100, each dollar that your stock goes down in value, your put pays you one dollar. Therefore, when you own this put in combination with the stock, the overall value of this combination cannot go below $100.

## Slide 90: Analogy with Car Insurance and Put/Call

- Understand the loss for Long and Short Position
- Let’s say that you cannot risk losing more than $10 while trading. There you want to be insured against the risk involved with holding a long/short position
- Therefore, we will opt for an ‘insurance’ for our stock position!
- Think about this in terms of Car Insurance and the associated deductibles
- In this example, since you don’t want to lose more than $10, our ‘deductible’ = $10 and we bought the stock for $100
- For each dollar below $90 you lose a $1 but also the insurance company pays you $1, so you are even beyond that point.

## Slide 91: Put Options

- Put: Gives the holder the right, but not the obligation to sell (100 shares) the underlying stock in the future at a pre-specified price ( strike price)
- 100-110 = Deductible
- Insurance - payoff for below 100

<details><summary>Speaker notes</summary>

Diagrams are wrong!

</details>

## Slide 92: Puts and Calls - are just contracts

- Notice that all of this occurred without you ever owning the stock to begin with.
- These are just types of contracts - that get made into financial contracts that can be bought and sold

## Slide 93: Hedging - combine into a portfolio

- How do options hedge us against loss?
- Example: Put
- +
- =
- Different Y axis!

## Slide 94: Option Pricing  - we can look at data in Lecture Notebook

- Options have premiums
- Premiums compensate for the person bearing a risk to provide the ‘insurance’
- Example: Call (Gives the holder the right to buy the underlying stock in the future at a pre-specified price)
- Factors that determine the price of options:
- Strike Price: The lower the strike price, the more I am going to charge
- Time until Expiration: Longer the time period, the more I am going to charge
- Volatility: If a stocks moves a lot, the more I am going to charge
- Underlying Stock Price: The higher the price of underlying stock, the more I am going to charge

## Slide 95: Black Scholes

- Don’t get intimidated we won’t derive it
- It take the four factors
- 𝑆  is the price of the stock,
- 𝐾 is the strike of the call,
- 𝑟 is the risk-free interest rate,
- 𝑇 is time till expiration,
- 𝑁() is the normal CDF, which measures of variability

## Slide 96: Returns

- The return on a security has two components:
- The change in price of the security
- Any cash flows associated with the security (dividends for stock, coupons for bonds)
- Rate of return (r) for change in price over a time interval     as

## Slide 97: In the Lecture Notebook

- Download data for SPY and Nasdaq
- Calculate returns
- Need to know ticker symbol and date range
- Pandas - calculate returns from difference from last time period
- In the homework
- Download Data for other stocks

## Slide 98: Graphs for Returns

- Time Series graph of S&P 500                       Graph for S&P 500 Returns

## Slide 99: Yfinance

## Slide 100: QuantStats

## Slide 101: Option - Twitter

## Slide 102: Python for Trading Decal - Stat 198

