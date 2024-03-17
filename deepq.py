import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from main import ConnectFour, WIDTH, HEIGHT

import random

print(torch.__version__)

from rl import RLAgent

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        #Hidden size can be experimented with
        #first layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        #second layer
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.output_size = output_size

    def forward(self, x):
        x = x.flatten(start_dim = 0)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
   
# Q-network
input_size = WIDTH * HEIGHT
hidden_size = 128

# Number of possible actions
output_size = WIDTH
q_network = QNetwork(input_size, hidden_size, output_size)

nn_saved_path = 'q_network_checkpoint.pth'

if os.path.exists(nn_saved_path):
    # Load the model checkpoint
    q_network.load_state_dict(torch.load('q_network_checkpoint.pth'))
    q_network.eval()  # Set the model to evaluation mode if needed

# Using Adam optimizer to tune NN weights, with learning rate
optimizer = optim.Adam(q_network.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

rl_agent = RLAgent(q_network, optimizer, loss_fn)

game = ConnectFour()

# Training loop
num_episodes = 50000
evaluation_interval = 5000
evaluation_episodes = 100
for episode in range(num_episodes):
    #reset environment if game ends
    state = game.reset()  
    done = False
    while not done:
        #Convert the states into numbers so they can be input in the NN
        state = game.get_state()

        # Choose action based on epsilon-greedy policy
        action = rl_agent.choose_action(state)
        
        # Take action in the environment
        next_state, reward, done = game.place(action)

        # Update Q-network
        rl_agent.update(state, action, next_state, reward)

        actions = ConnectFour.possible_actions(next_state)
        if not actions:
            actions = ConnectFour.possible_actions(next_state)
        action = random.choice(list(actions))
        op_next, op_reward, done = game.place(action)
        #tie
        if done and op_reward != -1:
            rl_agent.update(state, action, next_state, 0.5)
        else:
            rl_agent.update(state, action, next_state, reward)
        
        # Move to next state
        state = next_state
    
    # Evaluation phase: evaluate every interval specified
    if episode % evaluation_interval == 0:
        total_wins = 0
        total_losses = 0
        total_draws = 0
        #evaluate the agent certain amount of games, playing random opponent
        for _ in range(evaluation_episodes):
            # Initialize a new game for evaluation
            state = game.reset()
            done = False
            while not done:
                state = game.get_state()
                # Agent's turn
                action = rl_agent.choose_action(state)
                try:
                    next_state, reward, done = game.place(action)
                except ValueError:
                    action = rl_agent.choose_action(state)

                if done:
                    break

                #Random pponent's turn
                actions = ConnectFour.possible_actions(next_state)
                opponent_action = random.choice(list(actions))
                next_state, reward, done = game.place(opponent_action)

            #if agent won increment score
            if reward == 1.0:
                total_wins += 1
            elif reward == -1.0:
                total_losses += 1
            else:
                total_draws += 1

        # Calculate average win and loss
        win_rate = total_wins / evaluation_episodes
        loss_rate = total_losses / evaluation_episodes
        draw_rate = total_draws / evaluation_episodes
        print(f"Episode {episode}: Win Rate = {win_rate}, Loss Rate = {loss_rate}, Draw Rate = {draw_rate}")

#save the NN
torch.save(q_network.state_dict(), 'q_network_checkpoint.pth')
