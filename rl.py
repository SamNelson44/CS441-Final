from main import ConnectFour, WIDTH, HEIGHT, EMPTY
from collections import defaultdict
from copy import deepcopy
import numpy as np
import random
import time
from random import shuffle

GOAL = 3
NUM_PLAYERS = 2
CELL_VALUES = NUM_PLAYERS + 1 # includes empty cell
STATE_SIZE = HEIGHT * WIDTH * (NUM_PLAYERS + 1)
ACTION_SIZE = WIDTH

# Slightly modified State class based on Thanh's
class State(ConnectFour):
    def __init__(self):
        super().__init__()
        self.winner = None
        self.chips_placed = 0
        self.inverted = False

    def reset_board(self):
        self.board = [[None]*WIDTH for i in range(HEIGHT)]
        self.chips_size = [0]* WIDTH
        self.winner = None
        self.chips_placed = 0
        self.inverted = False

    def next_possible_moves(self):
        actions = []

        #if there is an empty spot at the top, we can drop chips at the column
        for column in range(WIDTH):
            if self.chips_size[column] <= HEIGHT:
                actions.append(column)
        return actions

    def print_board(self):
        for row in self.board:
            print('|', end='')
            for cell in row:
                if cell == 0:
                    print(' ', end='|')
                elif cell == 1:
                    print('X', end='|')
                else:
                    print('O', end='|')
            print()
        print('---------------')

    def place(self, column):
        """
        place a chip for self.current_player
        """
        result = False
        if column < 0 or column >= WIDTH:
            print(f'Invalid column :{column}', '\n')
        elif self.chips_size[column] >= HEIGHT:
            print(f'Too many pieces at column {column}', '\n')
        else:
            # Check which row to add the chip using chips_size at column
            # Add at the end of the array
            row = (HEIGHT - 1) - self.chips_size[column]
            self.board[row][column] = self.current_player

            # Increase chip size for the column
            self.chips_size[column] += 1
            self.switch_player()
            self.chips_placed +=1 # Board full counter
            result = True
        return result
    
    def inverted_board(self):
        tmp = deepcopy(self.board)
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if tmp[i][j] and tmp[i][j] == 'X':
                    tmp[i][j] = 'O'
                elif tmp[i][j] and tmp[i][j] == 'O':
                    tmp[i][j] = 'X'
        self.inverted = not self.inverted
        return tmp
    
    def check_win(self, i, j):
        if self.board[i][j] == EMPTY:
            return False
        
        # Up, right, up-right diagonal, down-right diagonal
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        result = False

        for x, y in directions:
            consecutive_count = 1 #current token
            token = self.board[i][j]

            # For each direction, check the direction and its opposite
            for dir in [1, -1]:
                next_i = i + x * dir
                next_j = j + y * dir

                # Move in direction, until token mismatch, or bound is reached
                while (next_i > 0 and  next_i < WIDTH-1
                       and next_j > 0 and  next_j < HEIGHT-1
                       and self.board[next_i][next_j] == token and consecutive_count < GOAL):
                    
                    consecutive_count += 1
                    next_i = i + x * dir
                    next_j = j + y * dir
                
                # Win condition met
                if consecutive_count >= GOAL:
                    result = True
                    self.winner = self.board[i][j]
        return result

    def board_full(self):
        return self.chips_placed >= HEIGHT * WIDTH

    def get_winner(self):
        return self.winner


# Define the Q-learning agent
class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        # Initialize the Q-table or neural network, idk which one yet

        self.q_values = np.zeros((STATE_SIZE, ACTION_SIZE))
        
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Initialize neural network
       # self.model = build_neural_network(STATE_SIZE, ACTION_SIZE)
    
    # def state_to_index(self, state):
    #     state_str = ''.join(str(cell) for row in state.get_state() for cell in row)
    #     row = hash(state_str) % STATE_SIZE
    #     col = (row % ACTION_SIZE)
    #     return row, col
    
    # 
    def state_to_index(self, state):
        index = 0
        for i in range(HEIGHT):
            for j in range(WIDTH):
                if state[i][j] == 'X':
                    index = index * (CELL_VALUES) + 1
                elif state[i][j] == 'O':
                    index = index * (CELL_VALUES) + 2
                else:
                    index = index * (CELL_VALUES)
        return index % STATE_SIZE

    def random_action(self, valid_actions):
        return random.choice(valid_actions)
    
    # def choose_action(self, state, valid_actions):
    #     print("valid actions:", valid_actions, "\n")
    #     # Epsilon-greedy
    #     if random.random() < self.epsilon:
    #         return self.random_action(valid_actions)
    #     else:
    #         state_index = self.state_to_index(state.get_state()) 
    #         return np.argmax(self.q_values[state_index, valid_actions])
    def choose_action(self, state, valid_actions):
        result = -1
        if random.random() < self.learning_rate:
            result = random.choice(valid_actions)  # Exploration
        else:
            q_values = []
            for action in valid_actions:
             state_index = self.state_to_index(state.get_state())
             q_values.append(self.get_q_value(state_index, action))

            max_q_value = max(q_values)

            top_choices = []
            for i in range(len(valid_actions)):
                if q_values[i] == max_q_value:
                    top_choices.append(i)
            result = valid_actions[random.choice(top_choices)]

        return result

    def get_reward(self, state):
        result = 0.0
        if state.get_winner() == 'X':
            result = 1  # Player wins
        elif state.get_winner == None and state.board_full():
            result = 0.5  # Draw
            
        return result  # Loss, or game is not done yet

    def get_q_value(self, state_index, action):
        # Get the Q-value for a given state-action pair
        return self.q_values[state_index, action]
    
    def update_q_values(self, state_index, action, reward, next_state_index):
        # Get best possible move
        max_next_q_value = max(self.q_values[next_state_index])
        
        # update Q-value
        new_q_value = self.q_values[state_index, action] + self.learning_rate * (reward + self.discount_factor * max_next_q_value - self.get_q_value(state_index, action))
        self.q_values[state_index, action] = new_q_value
    

    def train(self, epochs, random):
        state = State()
        # Train the agent using Q-learning
        for i in range(epochs):
            state.reset_board()
            done = False
            while not done:
                valid_actions = state.next_possible_moves()
                valid_move = False
                while not valid_move: 
                    if random and state.current_player == 'O':
                        action = self.random_action(valid_actions)
                    else:
                        action = self.choose_action(state, valid_actions)

                    state_index = self.state_to_index(state.get_state())

                    # Execute action and get next state
                    next_state = deepcopy(state)
                    valid_move = next_state.place(action)
                
                # Is game done?
                done = state.check_win(state.chips_size[action]-1, action)

                next_state_index = self.state_to_index(next_state.get_state())

                reward = self.get_reward(state)

                # Update Q-values
                self.update_q_values(state_index, action, reward, next_state_index)

                # Move to the next state
                state = next_state
            print("Game: ", i)    

def train_agent(epochs, learning_rate=0.1, discount_factor=0.90, epsilon=0.1, random=True):
    agent = QLearningAgent(learning_rate=0.3, discount_factor=0.90, epsilon=0.15)
    
    # Train the agent
    agent.train(epochs, random)
    return agent
    


def evaluate_agent(agent, total_games):
    state = State()
    
    # Get agent win%
    wins = 0
    draws = 0
        
    for i in range(total_games):
        state.reset_board()
        done = False
        for i in range(total_games):
            state.reset_board()
            done = False
            while not done:
                valid_actions = state.next_possible_moves()
                valid_move = False
                while not valid_move: 
                    if state.current_player == 'O':
                        action = agent.random_action(valid_actions)
                    else:
                        action = agent.choose_action(state, valid_actions)
                    
                    print(f'Coordinates :{state.chips_size[action]},{action}', '\n')

                    # Execute action and get next state
                    valid_move = state.place(action)
                
                # Is game done?
                done = state.check_win(state.chips_size[action]-1, action)
                reward = agent.get_reward(state)

                if done:
                    state.__str__()
                    reward = agent.get_reward(state)
                    if reward >= 1.0:
                        wins +=1
                    elif reward >= 0.4:
                        draws += 1
    win_rate = wins / total_games
    print("Win rate:", win_rate)

    draw_rate = draws / total_games
    print("Draw rate:", draw_rate)
