from main import ConnectFour, WIDTH, HEIGHT, EMPTY
from collections import defaultdict
from copy import deepcopy
import numpy as np
import random
import time
from random import shuffle

GOAL = 3
NUM_PLAYERS = 2
STATE_SIZE = HEIGHT * WIDTH * (NUM_PLAYERS + 1)
ACTION_SIZE = HEIGHT * WIDTH

# Slightly modified State class based on Thanh's
class State(ConnectFour):
    def __init__(self):
        super().__init__()
        self.winner = None

    def reset_board(self):
        self.board = [[None]*WIDTH for i in range(HEIGHT)]
        self.chips_size = [0]* WIDTH

    def inverted_board(self):
        tmp = deepcopy(self.board)
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if tmp[i][j] and tmp[i][j] == 'X':
                    tmp[i][j] = 'O'
                elif tmp[i][j] and tmp[i][j] == 'O':
                    tmp[i][j] = 'X'
        return tmp
    
    def check_win(self, i, j):
        if self.board[i][j] == EMPTY:
            return False
        
        #Up, right, up-right diagonal, down-right diagonal
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        result = False

        for x, y in directions:
            consecutive_count = 1 #current token
            token = self.board[i][j]

            #For each direction, check the direction and its opposite
            for dir in [1, -1]:
                next_i = i + x * dir
                next_j = j + y * dir

                #Move in direction, until token mismatch, or bound is reached
                while (next_i > 0 and  next_i < WIDTH-1
                       and next_j > 0 and  next_j < HEIGHT-1
                       and self.board[next_i][next_j] == token and consecutive_count < GOAL):
                    
                    consecutive_count += 1
                    next_i = i + x * dir
                    next_j = j + y * dir
                
                if consecutive_count >= GOAL:
                    result = True
        return result



    def get_winner(self):
        return self.winner


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
            valid_actions = state.possible_actions()
            action = agent.choose_action(state, valid_actions)
            next_state, reward, done = state.place(action)
            if done and reward == 1:
                wins += 1
    win_rate = wins / total_games
    print("Win rate:", win_rate)
