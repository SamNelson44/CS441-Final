from rl import *



# Training the agent
epochs = 40000
agent = train_agent(epochs, learning_rate=0.2, discount_factor=0.95, epsilon=0.7, random=True)


# Testing the agent
num_games = 100
evaluate_agent(agent, num_games)
