---
title: "Lecture 9 - Game Theory\_"
type: slides
week: 9
source_path: "/Users/ericvandusen/Documents/Data88E-ForTraining/F24LS/Lecture 9 - Game Theory_.pptx"
---

## Slide 1: Data 88E: Economic Models

- Lecture 9: Game Theory

## Slide 2: Announcements

- Assignments
  - Project 3 will be released
    - Its hard and complicated, don't snooze on it
    - Use the textbook!
  - Lab 8 will be released and will be due October 29
- https://cdss.berkeley.edu/dsus/advising-week
- Econ 148
- https://classes.berkeley.edu/content/2025-spring-econ-148-001-lec-001

## Slide 3: https://www.econ148.org/sp24/

## Slide 4: Nobel Prize in Economics 2024

## Slide 5: Nobel Prize in Economics 2024

- Growth - Which countries become wealthy
- Colonization
- Extractive Institutions  - just benefited those in power
- Inclusive Institutions - become more prosperous overall
- Democracy Matters - for long term Economic Growth

## Slide 6: Nobel Prize in Economics 2024

- Acemoglu and Johnson 2023
- To achieve the true potential of innovation, we need to ensure technology is creating new jobs and opportunities rather than marginalizing most people, through automated work and political passivity.  We need to use the tremendous digital advances of the last half century to create useful and empowering tools, and seize back control from a small elite of hubristic, messianic tech leaders pursuing
- their own interests.

## Slide 7: We used to run this as a skit in class

- We used to run this as a skit in class

## Slide 8: Silent         “Rat out”

( -1, -1)       ( -20,0)

( 0, -20 )      ( -10, -10)

- Silent         “Rat out”
- ( -1, -1)       ( -20,0)
- ( 0, -20 )      ( -10, -10)
- We used to run this as a skit in class
- Silent
- “Rat out”

## Slide 9: Lecture 8 Outline

- Game Theory Overview
- Strategy
- Prisoner’s Dilemma
- Nash Equilibria
- Oligopolies
- Cournot Competition
- Bertrand Competition
- Iterated Prisoner’s Dilemma
- A Healthy Dose of Irony: Webcam Placement Study

## Slide 10: Branch of mathematics concerned with the strategic interaction of rational decision-makers
Different types of games:
cooperative vs. non-cooperative
symmetric vs. asymmetric
simultaneous vs. sequential
perfect vs. imperfect information
We will study simultaneous games of perfect information.

- Branch of mathematics concerned with the strategic interaction of rational decision-makers
- Different types of games:
  - cooperative vs. non-cooperative
  - symmetric vs. asymmetric
  - simultaneous vs. sequential
  - perfect vs. imperfect information
- We will study simultaneous games of perfect information.
- Game Theory

## Slide 11: Cooperative vs Non-Cooperative

- Players = individual agents
- Coalitions = groups of players
- Payoffs =  individual payoffs or shares of a total payout
- Binding Agreements = some enforcement of payoff shares
- Prisoner’s Dilemma - non-cooperative Game
- Labor Union negotiations?

## Slide 12: C02 Emissions?

- Climate Change affects all
- Droughts, Hurricanes, Wildfires
- Economics Cost of limiting fossil fuels felt locally
- Incentive to Defect in Prisoner’s Dilemma
- https://ourworldindata.org/co2-and-greenhouse-gas-emissions

## Slide 13: (untitled)

## Slide 14: Kyoto Protocol ~  Paris Agreement

- Measure - Take Stock
- Plan to Reduce

## Slide 15: Symmetric vs Asymmetric

- Both players have the same information, rules , strategy
- Both players have same outcome set
- Chess - symmetric
- Poker - players have different hands -
- part of play is revealing about other players hands
- Sports - different strengths, strategies
- home field advantage
- In a game theory payoff matrix
- Values are the same for each player
- Symmetry is often a simplifying assumption

## Slide 16: Simultaneous vs Non-Simultaneous

- Both players are making a simultaneous choice unknown to the other
- Prisoners Dilemma - each are in a separate room making a deal
- Rock-Paper-Scissors - making a simultaneous choice
- Non - Simultaneous
- First Mover advantage  ( move to block other players)
- Second Mover advantage  ( move to unseat incumbent)

## Slide 17: Perfect vs Imperfect Information

- Perfect Information
- Transparency - both sides know the rules
- No Uncertainty - outcomes are certain
- No Hidden information - both sides may know each others capacity
- Imperfect information
- Rules, setup of game not known
- Uncertainty - chance outcomes
- Hidden information - other players position

## Slide 18: Aside - Bots taking over Online Poker

- https://www.bloomberg.com/features/2024-poker-bots-artificial-intelligence-russia

## Slide 19: Strategies

- Systematic methods of playing games
  - tell players what move to make based on available information
  - essentially a probability distribution over a player’s choices
- Different types
  - pure strategies put all probability on a single choice
  - mixed strategies divide probabilities over an array of choices
- Strategies can be random but often aren’t
  - expected utility is a common counter-strategy for randomness
  - Nash equilibria (if they exist) tell us the “best” move to make in games of imperfect information
- Project - Program Strategies to play a series of games

## Slide 20: Aside: Payoff Matrices - phase 1 enrollment

- Alan is trying to schedule his classes and wants to enroll in two classes with conflicting final exam times. One class will not give accommodations and is required for his degree. Alan knows the other class will give an accommodation with probability 0.6. Consider the payoff matrix below.

## Slide 21: Expected Utility

- Enroll in both               0.6 \* 10 + 0.4 \* -10 = 2
- Don’t enroll in both     0.6\* 2 + 0.4 \*5 = 3.2

## Slide 22: Expected Utility Hypothesis

- The expected utility of a set of n outcomes xi is the average utility of each outcome u(xi) weighted by the probability of that outcome’s occurrence pi:
- The expected utility hypothesis says that when an individual makes a gamble, they will choose the option that maximizes their expected utility
  - the utility of each option is based on their preferences
- Expected utility maximization (under the EUH) is one way of strategizing in the face of randomness

## Slide 23: Prisoner’s Dilemma

- Prolific game first considered in the 1950s
- Single-round symmetric simultaneous non-cooperative game of perfect information
- Each prisoner can either cooperate (maintain their silence) or defect (talk to the police)
  - Does this sound familiar?
- Reading the payoff matrix: a higher value (more years in prison) is worse! our goal is to minimize the values in the payoff matrix

## Slide 24: Nash Equilibrium

- A set of strategy choices in a non-cooperative game in which no player can increase their payoff by unilaterally changing their strategy.

## Slide 25: Nash Equilibrium

- A set of strategy choices in a non-cooperative game in which no player can increase their payoff by unilaterally changing their strategy.
- What about the prisoner’s dilemma?

<details><summary>Speaker notes</summary>

Annotate

</details>

## Slide 26: Nash Equilibrium

- A set of strategy choices in a non-cooperative game in which no player can increase their payoff by unilaterally changing their strategy.
- What about the prisoner’s dilemma?
- Nash equilibria can be read from payoff matrices: to see if a game state is a Nash equilibrium, the first value in the cell must be the maximum among all first values in that column and the second value must be the maximum across all second values in that row.

## Slide 27: Oligopolies

- Markets in which the number of participants is limited, e.g. airlines
- Allow their participants to function collectively in a manner similar to a monopoly
  - when this collective monopoly is illegal, these are referred to as cartels

## Slide 28: Oligopolies

- Markets in which the number of participants is limited, e.g. airlines
- Allow their participants to function collectively in a manner similar to a monopoly
  - when this collective monopoly is illegal, these are referred to as cartels
- Within oligopolies, however, we can observe competition
  - participants try to take market share by lowering prices below the agreed-upon floor

## Slide 29: Oligopolies

- Within oligopolies, however, we can observe competition
  - participants try to take market share by lowering prices below the agreed-upon floor
  - e.g. 2020 oil price war that was started when Saudi Arabia lowered its prices below the OPEC price in response to Russia refusing to reduce production

## Slide 30: Opec Market Share over time

## Slide 31: US enters the stage as top producer and large exporter

## Slide 32: Cournot Competition

- Model of a market in which firms compete by changing their output
  - market has a fixed number of firms that produce the same product
  - firms do not collude but have market power
  - each firm knows the number of firms and has its own cost function used to determine its level of output
  - firms share a single, fixed marginal cost
- Monopoly
- downward sloping Demand curve
- Marginal revenue downward sloping
- Homogeneous project
- No product differentiation

## Slide 33: Cournot Competition

- OPEC is a good example of a Cournot oligopoly
  - its members affect market prices by changing their level of production
  - but this shows a flaw: Cournot equilibrium implies that collusion is the rational policy, but game theory shows us that cartel members undercut one-another
  - 40% of global production    60% of exports

## Slide 34: (untitled)

## Slide 35: What are we trying to do?

- We want to find the quantities that maximize firm 1 and 2’s profit
- Recall from the supply and demand units that firms maximize profits when they produce where marginal cost equals marginal revenue.
- However, the marginal revenue changes depending on the quantity produced.

## Slide 36: What do these Best Response Functions Tell Us?

- This graph depicts two cases; one where firm 2 produces so little that firm 1’s best response is to become a monopoly and overtake the market, the other is where firm 2 produces so much that firm 1’s best response is to produce 0.
- The best response functions are great because they give us a profit-maximizing output for every possible quantity the other firm can produce at.
- However, a best response function is not the Nash Equilibrium.

## Slide 37: Finding the Nash Equilibrium - Python Sympy

- Another way to look at Nash Equilibria is both players mutually best responding to each other.
- In the context of this game this is where both player’s best response functions intersect.

## Slide 38: Aside on finding the Nash Equilibrium Algebraically

- This math is not necessary for this class, but it may be useful to your own understanding.

## Slide 39: Cournot

- Each firm knows market demand but doesn’t know others quantity
- Each firm considers the reaction of others when it set own quantity
- Oligopoly - lower quantities and higher prices than perfect competition
- Nash Equilibrium - when each firm has no incentive to change output

## Slide 40: Aside - Meat packing

- I brought this up in Demand Lecture 2
- Four companies control 80% Beef and 65% Pork
- Whether there is a cartel is a huge lawsuit right now

## Slide 41: USDA

- Reduce purchases of beef cattle
- Coordinate Purchases
- Idle plants, reduce capacity
- https://www.choicesmagazine.org/choices-magazine/submitted-articles/is-there-price-fixing-in-the-us-beef-packing-industry

## Slide 42: Collusion - Oligopoly and Data - Information Sharing

- The availability of price and other market information for industry participants facilitates transparent price discovery and increases market efficiency, which generally has substantial procompetitive effects, with no harm to competition involved. However, in the case of imperfectly competitive industries, publicly available price and other market information may facilitate information exchanges among competitors, some of which may have anticompetitive effects (Bloom, 2014). The firms with market power can use price and other market information to facilitate their tacit or overt collusion and to effectively enforce it. Tacit collusion is a coordinated conduct of firms with market power that does not involve an explicit agreement among them in violation of Section 1 of the Sherman Act. Overt collusion is a coordinated conduct of firms with marketpower that involves an explicit agreement among them, which would violate Section 1 of the Sherman Act.
- There are opinions expressed in academic literature that may suggest that the Livestock Mandatory Reporting Act might have created opportunities for tacit collusion in the beef packing industry (Wachenheim and DeVuyst, 2001; Cai, Stiegert, and Koontz 2011). In addition, some livestock producers expressed concerns that packers manipulated the USDA Agricultural Marketing Service Livestock Mandatory Reporting system (U.S. Department of Justice, 2012).
- Billions of dollars - explicit vs tacit collusion

## Slide 43: Bertrand Competition

- Similar to Cournot, but instead firms compete by changing price rather than output
  - assumes consumers want to pay the lowest price for goods
  - they will buy everything at the lower price
    - assumes that the firm offering the lower price has the capacity to meet the demand of the entire market at that price
    - if prices are equal, demand will be split evenly among them
  - predicts that even small competition (e.g. a duopoly) will result in prices being reduced to the marginal cost level

## Slide 44: Bertrand Competition

- Similar to Cournot, but instead firms compete by changing price rather than output
  - assumes consumers want to pay the lowest price for goods
  - they will buy everything at the lower price
    - assumes that the firm offering the lower price has the capacity to meet the demand of the entire market at that price
    - if prices are equal, demand will be split evenly among them
  - predicts that even small competition (e.g. a duopoly) will result in prices being reduced to the marginal cost level
- Coke vs. Pepsi is a good example of a Bertrand duopoly
  - both firms compete on price and take into account the price of their competitor
  - Defection benefits consumer

## Slide 45: Bertrand Competition

- Implications:
  - even a duopoly in a market is enough competition to push prices down to the level of perfect competition
  - but this model relies on serious assumptions
    - there are many reasons why consumers might not buy the lowest-priced item (e.g. non-price competition, search costs)
    - ignores the fact that firms may not be able to supply the entire market
  - including these factors results in a different conclusion

## Slide 46: Bertrand Competition

- Implications:
  - even a duopoly in a market is enough competition to push prices down to the level of perfect competition
  - but this model relies on serious assumptions
    - there are many reasons why consumers might not buy the lowest-priced item (e.g. non-price competition, search costs)
    - ignores the fact that firms may not be able to supply the entire market
  - including these factors results in a different conclusion
  - the model also demonstrates big incentives to cooperate and raise prices to the monopoly level
    - but this is not the Nash equilibrium of the model

## Slide 47: Firm 1 - setting price given P2

- Monopoly Price
- Cost of Production  - eg perf Comp price
- h= the cost to undercut
- (and get customers to switch)

## Slide 48: Bertrand - both firms reaction function

- Nash Equilibrium at Marginal Cost
- Because of perfect competition

## Slide 49: Bertrand Competition

## Slide 50: Bertrand Competition

## Slide 51: Iterated Prisoner’s Dilemma

- Paradigm in which the prisoner’s dilemma is played repeatedly over multiple rounds
- Players know the history of their opponent but not their current move
- Was used to model the Cold War and MAD in the 1950’s
  - different strategies for playing this game were analyzed in order to see which was most effective at winning the game
- A famous tournament for the Iterated Prisoner’s Dilemma was run by Robert Axelrod
  - the winning strategy at this tournament was tit-for-tat
  - excellent episode of Radiolab on this tournament

## Slide 52: Podcast - History of mathematical approaches to PD

## Slide 53: Play this game to learn about strategies

- https://ncase.me/trust

## Slide 54: Python - Strategies and Iterated Games

- Python
- Build a Strategy for a player to  play over a series of games
- Build a tournament
- Run a tournament with competing strategies
- Tit-For-Tat, Forgiver, Backstabber, Fool-me-once

## Slide 55: (untitled)

## Slide 56: Project 3

- In Project 3, you will
- study the iterated prisoner’s dilemma and recreate Axelrod’s tournament
- study an experiment by Thomas and Pemstein (2015) relating the perception of height to game strategies
- recreate the analysis of Thomas and Pemstein (2015) using A/B testing
  - webcam placement influences leader-follower behavior in games

## Slide 57: Web Camera Angle

- https://www.frontiersin.org/articles/10.3389/fpsyg.2015.00306/full\#F1

## Slide 58: Payoff Strategy in Webcam Game

## Slide 59: Project 3

- Learn about Python Classes in the Textbook
- players.py

## Slide 60: Lecture Notebook

- Play out the game that you will program into a tournament

## Slide 61: Lab - Tragedy of the Commons

- What is the tragedy of the commons?

## Slide 62: Tragedy of the Commons  - Lloyd

- Lloyd 1833 - Over grazing of common Lands
- Each herder can add one more cattle
- But overall leads to degradation of pasture
- How does this represent a multi-player game?

## Slide 63: Tragedy of the Commons - Hardin 1968

- Human Overpopulation
- Each family has an incentive to have more kids
- Overpopulation leads to environmental disaster
- Welfare State enables families to avoid consequences of large families
- Very influential article
- Biology -> Competition

## Slide 64: Tragedy of the Commons - Elinor Ostrom  - Nobel 2009

- Hardin was wrong
- Common Property Resources
- Society figures out how to control over exploitation of resources
- Groundwater in California
- Grazing in Switzerland
- Fishing in Maine
- Irrigation in Spain
- Multiple Equilibria are possible
- Institutions Matter

