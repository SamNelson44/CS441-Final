import numpy as np
import random
#import matplotlib.pyplot as plt

class ConnectFour:
    def __init__(self):
        self.board = np.zeros((6, 7))  # 6 rows, 7 columns
        self.current_player = 1  # Player 1 starts

    def reset(self):
        self.board = np.zeros((6, 7))
        self.current_player = 1

    def is_valid_move(self, column):
        return self.board[0][column] == 0

    def make_move(self, column):
        for row in range(5, -1, -1):
            if self.board[row][column] == 0:
                self.board[row][column] = self.current_player
                break
        self.current_player = 3 - self.current_player  # Switch players

    def get_winner(self):
        # Check rows
        for row in range(6):
            for col in range(4):
                if self.board[row][col] != 0 and \
                   self.board[row][col] == self.board[row][col+1] == \
                   self.board[row][col+2] == self.board[row][col+3]:
                    return self.board[row][col]

        # Check columns
        for col in range(7):
            for row in range(3):
                if self.board[row][col] != 0 and \
                   self.board[row][col] == self.board[row+1][col] == \
                   self.board[row+2][col] == self.board[row+3][col]:
                    return self.board[row][col]

        # Check diagonals (positive slope)
        for row in range(3):
            for col in range(4):
                if self.board[row][col] != 0 and \
                   self.board[row][col] == self.board[row+1][col+1] == \
                   self.board[row+2][col+2] == self.board[row+3][col+3]:
                    return self.board[row][col]

        # Check diagonals (negative slope)
        for row in range(3):
            for col in range(3, 7):
                if self.board[row][col] != 0 and \
                   self.board[row][col] == self.board[row+1][col-1] == \
                   self.board[row+2][col-2] == self.board[row+3][col-3]:
                    return self.board[row][col]

        # Check for draw
        if not any(0 in row for row in self.board):
            return -1  # Draw

        return 0  # Game still ongoing

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


###########################################################
################ Q-learning Class #########################
###########################################################


class QLearningAgent:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=1.0):
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.q_table = {}       # Q-value table

    def get_action(self, state, valid_moves):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_moves)  # Exploration
        else:
            # Exploitation
            state_str = ''.join(str(int(cell)) for row in state for cell in row)
            if state_str not in self.q_table:
                self.q_table[state_str] = [0] * 7  # Initialize Q-values for this state
            # Choose action with maximum Q-value
            max_q_value = max([self.q_table[state_str][col] for col in valid_moves])
            best_actions = [col for col in valid_moves if self.q_table[state_str][col] == max_q_value]
            return random.choice(best_actions)

    def update_q_value(self, state, action, reward, next_state):
        state_str = ''.join(str(int(cell)) for row in state for cell in row)
        next_state_str = ''.join(str(int(cell)) for row in next_state for cell in row)
        if state_str not in self.q_table:
            self.q_table[state_str] = [0] * 7  # Initialize Q-values for this state
        if next_state_str not in self.q_table:
            self.q_table[next_state_str] = [0] * 7  # Initialize Q-values for next state
        max_next_q_value = max(self.q_table[next_state_str])
        self.q_table[state_str][action] += self.alpha * \
            (reward + self.gamma * max_next_q_value - self.q_table[state_str][action])

def play_game(agent, env):
    env.reset()
    while True:
        state = env.board.copy()
        valid_moves = [col for col in range(7) if env.is_valid_move(col)]
        action = agent.get_action(state, valid_moves)
        env.make_move(action)
        next_state = env.board.copy()
        winner = env.get_winner()
        if winner != 0:
            if winner == -1:
                reward = 0.5  # Draw
            elif winner == 1:
                reward = 1.0  # Player 1 wins
            else:
                reward = 0.0  # Player 2 wins
            agent.update_q_value(state, action, reward, next_state)
            return winner
        else:
            reward = 0.0
            agent.update_q_value(state, action, reward, next_state)

def test_agent(agent, env, num_games=10):
    wins = 0
    for _ in range(num_games):
        winner = play_game(agent, env)
        if winner == 1:
            wins += 1
    return wins


###########################################################
######################### Main ############################
###########################################################


if __name__ == "__main__":
    env = ConnectFour()
    agent = QLearningAgent()

    # Training the agent
    num_epochs = 10000
    wins_over_time = []
    for epoch in range(1, num_epochs + 1):
        play_game(agent, env)
        if epoch % 1000 == 0:
            wins = test_agent(agent, env)
            wins_over_time.append(wins)
    print(wins_over_time)

    # Plotting the results
    '''
    plt.plot(range(1000, num_epochs + 1, 1000), wins_over_time)
    plt.xlabel('Epoch')
    plt.ylabel('Wins')
    plt.title('Wins over Time')
    plt.show()
    '''
    
