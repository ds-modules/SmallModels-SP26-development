---
title: "lecNB-prisoners-dilemma"
type: lecture-notebook
week: 9
source_path: "/Users/ericvandusen/Documents/Data88E-ForTraining/F24Lec_NBs/lec09/lecNB-prisoners-dilemma.ipynb"
---

<table style="width: 100%;" id="nb-header">
    <tr style="background-color: transparent;"><td>
        <img src="https://data-88e.github.io/assets/images/blue_text.png" width="250px" style="margin-left: 0;" />
    </td><td>
        <p style="text-align: right; font-size: 10pt;"><strong>Economic Models</strong>, EdX<br>
            Dr. Eric Van Dusen <br>
            Chris Pyles <br>
        Akhil Venkatesh <br>
</table>

```python
from players import *
```

# Lecture Notebook: Iterated Prisoner's Dilemma

## Iterated Prisoner's Dilemma

```python
def human_play(self, opponent):
    if len(opponent.history):
        print(f"Your opponents last move: {'D' if opponent.history.item(-1) else 'C'}")
    move = input("Your move: (d/c) ").strip().upper()
    assert move in ["D", "C"], f"invalid move: {move}"
    return move == "D"

Human = create_player_class("Human", human_play)
```

```python
n_rounds = input("How many rounds would you like to play? [10] ")
if n_rounds:
    n_rounds = int(n_rounds)
else:
    n_rounds = 10

show_opponent = input("Show opponent? (y/n) [n] ").strip().upper() == "Y"

human = Human()
opponent = np.random.choice(make_array(
    Alternator(), Backstabber(), Bully(), Desperate(), FoolMeOnce(), Forgiver(), 
    Grudger(), OnceBitten()
))

if show_opponent:
    print(f"Your opponent is: {opponent}")

winner = run_match(human, opponent, n_rounds)
if winner == Human():
    winner = "you!"
print(f"The winner is: {winner}")
```

### Using `create_player_class`

**From the project:**

In this project, we will be creating some custom Python classes to represent different playing strategies. We'll be using the `create_player_class` function (provided) to create these classes for us. Here are a few important things to know about the classes:

Consider a player class instance `player = Player()`. 
* `player.play()` is a method that represents a single move. It returns `True` if the player **defects** and `False` if they cooperate.
* `player.history` is an array of the previous moves of the player instance. `player.history.item(-1)`, for example, is the most recent move (the last element of the array).

Don't worry about the other methods that are defined for players; these are there for the code below to work but you don't need to concern yourself with them. Here's how the `create_player_class` function works:

1. Define a function, `f`, that takes two arguments: `self` and `opponent` (more about these later), and returns `True` if the player defects and `False` otherwise.
2. Call `create_player_class` with two arguments: a string for the class name, e.g. `"Player"`, and the play method `f`.

Let's illustrate this by creating `Defector`, a player that always defects. The `defector_play` function below always returns `True`, since the player always defects. We then create `Defector` using `create_player_class`.

**Example:** Defector

```python
def defector_play(self, opponent):
    return True

Defector = create_player_class("Defector", defector_play)
```



