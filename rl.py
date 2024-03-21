from main import ConnectFour, WIDTH, HEIGHT
from collections import defaultdict
from copy import deepcopy
import numpy as np
import random
import time
from random import shuffle

NUM_PLAYERS = 2
STATE_SIZE = HEIGHT * WIDTH * (NUM_PLAYERS + 1)
ACTION_SIZE = HEIGHT * WIDTH

# Slightly modified State class based on Thanh's
class State(ConnectFour):
    def __init__(self):
        super().__init__()

    def next_possible_moves(self):
        res = []
        for idx,chips in enumerate(self.chips_size):
            if chips < HEIGHT:
                res += [(HEIGHT-chips-1, idx)]
        return res
    
    def place(self, row, col):
        if self.board[row][col] == 'X' or self.board[row][col] == 'O':
            raise ValueError(f'Board position is already taken at {row},{col}?\n {self.board} \n{self.chips_size}\n{self.next_possible_moves()}')
        self.board[row][col] = self.current_player

        #increase chip size for the column
        self.chips_size[col] += 1
        self.switch_player()
    
    def inverted_board(self):
        tmp = deepcopy(self.board)
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if tmp[i][j] and tmp[i][j] == 'X':
                    tmp[i][j] = 'O'
                elif tmp[i][j] and tmp[i][j] == 'O':
                    tmp[i][j] = 'X'
        return tmp
    
    
    def reset_board(self):
        self.board = [[None]*WIDTH for i in range(HEIGHT)]
        self.chips_size = [0]* WIDTH


def build_neural_network(input_shape, output_shape):
    # Build a neural network model using TensorFlow/Keras
    pass


# Define the Q-learning agent
class QLearningAgent:
    def __init__(self, height, width, num_players, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        # Initialize the Q-table or neural network, idk which one yet

        self.q_values = np.zeros((STATE_SIZE, ACTION_SIZE))
        
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Initialize neural network
        self.model = build_neural_network(STATE_SIZE, ACTION_SIZE)
    

    def random_action(self, valid_actions):
        return random.choice(valid_actions)
    
    def choose_action(self, state, valid_actions):
        # Epsilon-greedy
        if random.random() < self.epsilon:
            return self.random_action(valid_actions)
        else:
            return np.argmax(self.q_values[state, valid_actions])

    # Thahn's     
    def check_status(self, i, j):
        def win(i, j):
            if self.state.current_player == 'X':
                record = self.count_lines(self.state.board, i, j)
            else:
                record = self.count_lines(self.state.inverted_board(), i, j)
            return 4 in record.keys()
        
        if win(i, j):
            if self.state.current_player == 'X':
                return 'X wins'
            else:
                return 'O wins'
        #if there is only one move left -> game over
        if sum(self.state.chips_size) == 41:
            print(self.state.chips_size)
            return 'Draw!'
        return None

    def get_reward(self, player):
        # Calculate and return the reward based on the game outcome
        if self.is_winner(player):
            return 1  # Player wins
        elif self.is_winner(3 - player):
            return -1  # Opponent wins
        elif self.is_draw():
            return 0.5  # Draw
        else:
            return 0  # Game still ongoing
    
    def update_q_values(self, state, action, reward, next_state):
        pass
    
    def train(self, epochs, state):
        # Train the agent using Q-learning
        pass


def train_agent(epochs, random=True):
    state = ConnectFour()
    agent = QLearningAgent(STATE_SIZE, ACTION_SIZE)
    
    # Train the agent
    agent.train(epochs, state)
    


def evaluate_agent(games):
    state = ConnectFour()
    agent = QLearningAgent(STATE_SIZE, ACTION_SIZE)
    
    # Get agent win%
    wins = 0
    total_games = games
        
    for i in range(total_games):
        state.reset_board()
        done = False
        while not done:
            valid_actions = state.next_possible_moves()
            action = agent.choose_action(state, valid_actions)
            next_state, reward, done = state.step(action)
            if done and reward == 1:
                wins += 1
    win_rate = wins / total_games
    print("Win rate:", win_rate)
