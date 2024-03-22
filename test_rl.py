from rl import *



# Training the agent
epochs = 10
agent = train_agent(epochs, random=True)


# Testing the agent
num_games = 10
evaluate_agent(agent, num_games)
