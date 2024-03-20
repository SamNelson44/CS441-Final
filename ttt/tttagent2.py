import numpy as np
import random
from collections import defaultdict
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class TicTacToeAgent:
    def __init__(self, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001, gamma=0.99):
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay  # Exploration decay rate
        self.learning_rate = learning_rate  # Learning rate
        self.gamma = gamma  # Discount factor
        self.q_values = defaultdict(lambda: np.zeros(9))  # Q-values initialized as zeros
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=9, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(9, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def choose_action(self, state):
        state= tuple(state)
        if np.random.rand() <= self.epsilon:
            return random.choice(self.available_actions(state))
        else:
            q_values = self.q_values[state]
            return np.argmax(q_values)

    def available_actions(self, state):
        return [i for i, val in enumerate(state) if val == 0]

    def update_q_values(self, state, action, reward, next_state):
        state_key = tuple(state)  # Convert state to a tuple
        q_values = self.q_values[state_key]
        
        next_state_key = tuple(next_state)
        next_q_values = self.q_values[next_state_key]
        max_next_q_value = np.max(next_q_values)
        new_q_value = reward + self.gamma * max_next_q_value
        q_values[action] = (1 - self.learning_rate) * q_values[action] + self.learning_rate * new_q_value
        self.q_values[state_key] = q_values


    def train(self, episodes):
        for episode in range(episodes):
            state = [0] * 9  # Initial state
            done = False
            while not done:
                action = self.choose_action(state)  # Pass state directly
                next_state, reward, done = self.make_move(state, action)
                self.update_q_values(state, action, reward, next_state)  # Pass state directly
                state = next_state
                if done:
                    break
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


    def make_move(self, state, action):
        next_state = state.copy()
        next_state[action] = 1  # Assuming the agent is 'X' and the opponent is 'O'
        if self.check_winner(next_state):
            reward = 1
            done = True
        elif self.check_draw(next_state):
            reward = 0.5
            done = True
        else:
            reward = 0
            done = False
        return next_state, reward, done

    def check_winner(self, state):
        # Check rows, columns, and diagonals for a winner
        winning_combinations = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
        for combo in winning_combinations:
            if state[combo[0]] == state[combo[1]] == state[combo[2]] == 1:
                return True
        return False

    def check_draw(self, state):
        return 0 not in state

    def play(self):
        state = [0] * 9
        done = False
        while not done:
            print("Current Board:")
            self.print_board(state)
            action = int(input("Choose your move (0-8): "))
            state, _, done = self.make_move(state, action)
            if done:
                print("Game Over!")
                self.print_board(state)
                if self.check_winner(state):
                    print("You win!")
                else:
                    print("It's a draw.")
                break
            print("Agent's move:")
            action = self.choose_action(str(state))
            state, _, done = self.make_move(state, action)
            if done:
                print("Game Over!")
                self.print_board(state)
                if self.check_winner(state):
                    print("Agent wins!")
                else:
                    print("It's a draw.")
                break

    def print_board(self, state):
        symbols = [' ', 'X', 'O']
        for i in range(3):
            print('-------------')
            print(f'| {symbols[state[i*3]]} | {symbols[state[i*3+1]]} | {symbols[state[i*3+2]]} |')
        print('-------------')

if __name__ == "__main__":
    agent = TicTacToeAgent()
    agent.train(episodes=10000)
    agent.play()
