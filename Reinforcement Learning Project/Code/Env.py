# In this project, the Environment is the Market
# where the Agent (the Chef) is trying to collect the ingredients needed to create high-valued recipes

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ChefEnv(gym.Env):
    def __init__(self, ):
        super().__init__()

        # 5x5 grid Market environment
        self.grid_size = 5
        self.state = (0, 0)  # starting from the top-left corner

        # Define the Action space (0: up, 1: down, 2: left, 3: right)
        self.action_space = spaces.Discrete(4)

        # Define the list of ingredients (to make pizza recipes)
        self.ingredient_list = ['dough', 'tomato', 'cheese', 'tuna', 'pepperoni', 'pineapple']
        self.valid_recipes = {
            ('dough', 'sauce', 'cheese'): 6,  # Margherita pizza
            ('dough', 'sauce', 'cheese', 'pepperoni'): 8,  # Pepperoni pizza
            ('dough', 'sauce', 'cheese', 'tuna'): 10,  # Tuna pizza
        }

        # Ingredient costs
        self.ingredient_costs = {'dough': 3, 'sauce': 3, 'cheese': 4, 'pepperoni': 5, 'tuna': 6, 'pineapple': 3}

        # Budget limit
        self.max_budget = 20

        # Cumulative earnings and Remaining budget across episodes
        self.cumulative_earnings = 0
        self.total_remaining_budget = 0

        # Define the Observation space (the agent's position, collected ingredients, budget)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.grid_size * self.grid_size),  # Position of the agent in the grid
             spaces.MultiBinary(len(self.ingredient_list)),  # Collected ingredients
             spaces.Discrete(self.max_budget + 1))  # Remaining budget
        )

        # Reset environment
        self._reset()

    def _reset(self):
        """
        Reset the environment to the initial state after each episode
        """
        self.state = (0,0)
        self.collected_ingredients = set()
        self.current_budget = self.max_budget
        return self._get_observation()

    def step(self, action):
        """
        The interaction between the Environment and the Agent
        """
        x, y = self.state

        if action == 0 and y > 0: # Up
            y -= 1
        elif action == 1 and y < self.grid_size - 1: # Down
            y += 1
        elif action == 2 and x > 0: # Left
            x -= 1
        elif action ==3 and x < self.grid_size- 1: # Right
            x += 1

        self.state = (x,y)

        # The ingredients are randomly assigned to the cells
        ingredient = self.ingredient_list[(x+y) % len(self.ingredient_list)]

        reward = 0
        done = False

        # If the agent picks pineapple, give penalty and end episode
        if ingredient == 'pineapple':
            reward -= 10 # The agent gets penalty
            done = True
        else:
            if ingredient not in self.collected_ingredients:
                cost = self.ingredient_costs[ingredient]
                if self.current_budget >= cost:
                    self.collected_ingredients.add(ingredient)
                    self.current_budget -= cost
                else:
                    reward -= 5 # The agent gets penalty for exceeding budget
                    done = True

            # Check if a valid recipe is made
            for recipe, value in self.valid_recipes.items():
                if set(recipe) == self.collected_ingredients:
                    reward = value
                    self.cumulative_earnings += value
                    self.total_remaining_budget += self.current_budget
                    done = True
                    break

            # Check if invalid ingredient combination
            if any(self.collected_ingredients - set(sum(self.valid_recipes.keys(), ()))):
                reward -= 5
                done = True

        return self._get_observation(),reward, done, {}

    def _get_observation(self):
        """
        Return the current observation (position, collected ingredients, budget)
        """
        x, y = self.state
        pos_index = y*self.grid_size + x
        collected_binary = [
            1 if ing in self.collected_ingredients
            else 0 for ing in self.ingredient_list
        ]
        return pos_index, tuple(collected_binary), self.current_budget

    def render(self):
        """Render the environment state."""
        grid = np.full((self.grid_size, self.grid_size), '_', dtype=str)
        x, y = self.state
        grid[y, x] = 'C'  # Mark agent position
        print("\n".join(" ".join(row) for row in grid))
        print(f"Collected: {self.collected_ingredients}, Budget: {self.current_budget}")

    def close(self):
        """Clean up resources if necessary."""
        pass