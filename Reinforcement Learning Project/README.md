## Reinforcement Learning Project
#### _Chi Hoang_
#### _Data Science for Economics - UNIMI_

### Introduction to the Environment
- In this game, the **Agent plays as a Chef** who explores the city markets, in order to find the best ingredients from a limited supply in order to makes delicious pizzas.
- The city map is represented as a **5x5 grid**, where each ingredient is randomly assigned to a cell in the grid.
- The Agent moves around the grid **(up, down, left, right)**, and collect the ingredients ('dough', 'tomato', 'pepperoni', 'tuna', 'pineapple')
- Each recipe is assigned to a particular Monetary value
- Each ingredient is assigned to a specific Cost (to compare with the Budget given to the Agent)
- An episode is ended when:
  - The Agent picks 'pineapple' **(Penalty)**
  - The Agent spends more than the Budget given **(Penalty)**
  - The Agent make invalid recipe **(Penalty)**
  - The Agent make valid recipe **(Reward = The remaining budget + The value of the recipe)**

### Introduction to the Agent
- Q-learning 
- SARSA
- Linear approximation



