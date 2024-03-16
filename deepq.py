import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from main import ConnectFour

import random

print(torch.__version__)

from rl import RLAgent

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.output_size = output_size


    def forward(self, x):
        x = x.flatten(start_dim = 0)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
   
# Initialize Q-network
input_size = 6 * 7  # Size of the game board
hidden_size = 128
output_size = 7  # Number of possible actions
q_network = QNetwork(input_size, hidden_size, output_size)

# Define optimizer and loss function
optimizer = optim.Adam(q_network.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

rl_agent = RLAgent(q_network, optimizer, loss_fn)


connect_four_env = ConnectFour()

# Training loop
num_episodes = 3000
evaluation_interval = 100
evaluation_episodes = 100
for episode in range(num_episodes):
    state = connect_four_env.reset()  # Reset environment to initial state
    done = False
    while not done:
        # Choose action based on epsilon-greedy policy
        state = connect_four_env.get_state()
        action = rl_agent.choose_action(state)
        
        # Take action in the environment
        next_state, reward, done = connect_four_env.place(action)

        # Update Q-network
        rl_agent.update(state, action, next_state, reward)

        actions = ConnectFour.possible_actions(next_state)
        if not actions:
            actions = ConnectFour.possible_actions(next_state)
        action = random.choice(list(actions))
        op_next, op_reward, done = connect_four_env.place(action)
        #tie
        if done and op_reward != -1:
            rl_agent.update(state, action, next_state, 0.5)
        else:
            rl_agent.update(state, action, next_state, reward)
        
        # Move to next state
        state = next_state
    
    # Evaluation phase
    if episode % evaluation_interval == 0:
        total_wins = 0
        total_losses = 0
        total_draws = 0
        for _ in range(evaluation_episodes):
            # Initialize a new game for evaluation
            state = connect_four_env.reset()
            done = False
            while not done:
                state = connect_four_env.get_state()
                # Agent's turn
                action = rl_agent.choose_action(state)
                try:
                    next_state, reward, done = connect_four_env.place(action)
                except ValueError:
                    action = rl_agent.choose_action(state)

                if done:
                    break

                # Opponent's turn (random player for simplicity)
                actions = ConnectFour.possible_actions(next_state)
                opponent_action = random.choice(list(actions))
                next_state, reward, done = connect_four_env.place(opponent_action)

            # Update evaluation metrics based on game outcome
            if reward == 1.0:
                total_wins += 1
            elif reward == -1.0:
                total_losses += 1
            else:
                total_draws += 1

        # Calculate and print evaluation metrics
        win_rate = total_wins / evaluation_episodes
        loss_rate = total_losses / evaluation_episodes
        draw_rate = total_draws / evaluation_episodes
        print(f"Episode {episode}: Win Rate = {win_rate}, Loss Rate = {loss_rate}, Draw Rate = {draw_rate}")


