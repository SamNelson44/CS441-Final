from rl import *



# Training the agent
epochs = 100000
agent = train_agent(epochs, random=True)


# Testing the agent
num_games = 100
evaluate_agent(agent, num_games)
