import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from game import TTT
WIDTH = 3
HEIGHT = 3

import random

print(torch.__version__)

from tttrl import RLAgent

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        #Hidden size can be experimented with
        #first layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        #second layer
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.output_size = output_size

    def forward(self, state, action):
        action_tensor = torch.tensor(action, dtype=torch.float32)
        x = torch.cat((state.flatten(), action_tensor), dim = 0)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Q-network
input_size = WIDTH * HEIGHT + 2
hidden_size = 8

# Number of possible actions (coordinates)
output_size = 1  
q_network = QNetwork(input_size, hidden_size, output_size)

# Using Adam optimizer to tune NN weights, with learning rate
optimizer = optim.Adam(q_network.parameters(), lr=0.003)
loss_fn = nn.MSELoss()

rl_agent = RLAgent(q_network, optimizer, loss_fn)

game = TTT()

# Training loop
num_episodes = 40000
evaluation_interval = 2000
evaluation_episodes = 100
for episode in range(num_episodes):
    #reset environment if game ends
    state = game.reset()  
    done = False
    while not done:
        # Choose action based on epsilon-greedy policy
        state = game.get_state()
        action = rl_agent.choose_action(state)
        
        # Take action in the environment
        next_state, done = game.place(action)
        reward = game.get_reward()
        if done:
            rl_agent.update(state, action, next_state, reward)
            break
       
        rl_agent.update(state, action, next_state, reward)

        # Update Q-network


        #Random opponent's turn
        actions = game.possible_actions(next_state)
        opponent_action = random.choice(list(actions))
        op_next , done = game.place(opponent_action)
        reward = game.get_reward()
        rl_agent.update(state, action, next_state, reward)
       
    
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
            reward = 0
            while not done:
                # Agent's turn
                state = game.get_state()

                action = rl_agent.choose_action(state)
                next_state, done = game.place(action)
                reward = game.get_reward()

                if done:
                    break

                #Random opponent's turn
                #player turn
                # print(game)
                actions = game.possible_actions(next_state)
                actions = list(actions)
                opponent_action = random.choice(actions)
                next_state, done = game.place(opponent_action)
                reward = game.get_reward()

                

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
