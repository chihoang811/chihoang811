import numpy as np
from abc import ABC, abstractmethod


class TabularAgent(ABC):
    def __init__(self, action_space,
                 gamma=0.9,  # discount factor
                 learning_rate=0.1,
                 epsilon=0.1
                 ):
        self.action_space = action_space
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # Initialize Q-table (for each state-action pair)
        self.Q = np.zeros((self.action_space,))

    @abstractmethod
    def update(self, state, action, reward, next_state, next_action):
        pass

    @abstractmethod
    def eplore(self, state):
        pass

    @abstractmethod
    def train(self, env, max_iterations=1000):
        pass


class QLearning(TabularAgent):
    def __init__(self, action_space,
                 gamma=0.9,
                 learning_rate=0.1,
                 epsilon=0.1
                 ):
        super().__init__(action_space, gamma, learning_rate, epsilon)

    def update(self, state, action, reward, next_state, next_action=None):
        best_next_action = np.argmax(self.Q[next_state])  # choose the action giving the best Q(s,a)
        self.Q[state, action] += self.learning_rate * (
                    reward + self.gamma * self.Q[next_state, best_next_action] - self.Q[state, action])

    def train(self, env, max_iterations=1000):
        for episode in range(max_iterations):
            state = env.reset()
            done = False
            while not done:
                action = self.explore(state)
                next_state, reward, done, _ = env.step(action)
                self.update(state, action, reward, next_state)
                state = next_state

    def explore(self, state):
        """
        Epsilon-greedy
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.Q[state])


class Sarsa(TabularAgent):
    def __init__(self, action_space, gamma=0.9,
                 learning_rate=0.1,
                 epsilon=0.1
                 ):
        super().__init__(action_space, gamma, learning_rate, epsilon)

    def update(self, state, action, reward, next_state, next_action):
        # SARSA update rule
        self.Q[state, action] += self.learning_rate * (
                    reward + self.gamma * self.Q[next_state, next_action] - self.Q[state, action])

    def train(self, env, max_iterations=1000):
        for episode in range(max_iterations):
            state = env.reset()
            action = self.explore(state)  # Select the first action
            done = False
            while not done:
                next_state, reward, done, _ = env.step(action)
                next_action = self.explore(next_state)  # Choose next action based on the policy
                self.update(state, action, reward, next_state, next_action)
                state, action = next_state, next_action

    def explore(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.Q[state])
