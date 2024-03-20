import torch
from game import TTT

import random


class RLAgent():
    def __init__(self, q_network, optimizer, loss_fn, gamma = 0.99):
        self.q_network = q_network
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.gamma = gamma #Discount reward
        self.epsilon = 0.2

    
    def update(self, old_state, action, new_state, reward):
        """
        Update Q(s,a) given a state action pair and a reward after the action
        """
        # Convert states to tensors
        old_state_tensor = torch.tensor(old_state, dtype=torch.float32)

        # Convert action to tensor
        action_tensor = torch.tensor(action, dtype=torch.float32).clone().detach().requires_grad_(True)

        # Get Q-value for the old state-action pair
        q_value_old = self.q_network(old_state_tensor.clone().detach(), action_tensor.clone().detach())

        # Calculate the maximum Q-value among all actions in the new state
        max_q_value_new = self.calculate_max_q_value(new_state)

        # Calculate the target Q-value using the Q-learning update rule
        target_q_value = reward + self.gamma * max_q_value_new
        target_q_value = q_value_old   + 0.5 * (target_q_value - q_value_old)
        target_q_value = torch.tensor(target_q_value, dtype=torch.float32)

 
        # Calculate the loss (mean squared error between predicted Q-value and target Q-value)
        loss = self.loss_fn(q_value_old, target_q_value)

        # Perform backpropagation and update the parameters of the neural network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def calculate_max_q_value(self,state):
        """
        Calculate the maximum Q-value for all possible actions in a given state
        """
        new_state_tensor = torch.tensor(state, dtype=torch.float32)
        max_q_value = 0
        
        for action in TTT.possible_actions(state):  # Iterate over all possible actions
            action_tensor = torch.tensor(action, dtype=torch.float32)
            q_value = self.q_network(new_state_tensor, action_tensor)
            max_q_value = max(max_q_value, q_value)
        return max_q_value

    def best_action(self, state):
        """
        Given a state get the maximum q-value we can get from all possible action
        """
        actions = TTT.possible_actions(state)
        highest_q = 0
        for action in actions:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_value = self.q_network(state_tensor, action)
            if q_value > highest_q:
                highest_q = q_value
        return highest_q

    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy policy.

        Returns:
            action: The chosen action.
        """

        state_tensor = torch.tensor(state, dtype=torch.float32)

        actions = TTT.possible_actions(state)
        actions = list(actions)

        # Explore with probability epsilon
        if random.random() < self.epsilon:
            # Randomly choose an action
            action = random.choice(actions)
        else:
            # Choose action with maximum Q-value
            for action in actions:
                with torch.no_grad():
                    q_values = self.q_network(state_tensor, action)
                    max_index = torch.argmax(q_values).item()
                    action = actions[max_index]

        if not action:
            self.choose_action(state)
        return action
       
    