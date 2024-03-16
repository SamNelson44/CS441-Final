import torch
from main import ConnectFour, WIDTH, HEIGHT, EMPTY
import random


class RLAgent():
    def __init__(self, q_network, optimizer, loss_fn, gamma = 0.99):
        self.q_network = q_network
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.gamma = gamma
        self.epsilon = 0.5

    
    def update(self, old_state, action, new_state, reward):
        """
        Update Q(s,a) given a state action pair and a reward after the action
        """
        if ConnectFour.is_board_empty(old_state) or ConnectFour.is_board_empty(new_state):
            return 

        # Convert states to tensors
        old_state_tensor = torch.tensor(old_state, dtype=torch.float32)
        new_state_tensor= torch.tensor(new_state, dtype=torch.float32)

        # Get Q-values for the old and new states
        q_values_old = self.q_network(old_state_tensor)
        q_values_new = self.q_network(new_state_tensor)

        # Calculate target Q-value using Bellman equation
        target_q_value = reward + self.gamma * q_values_new.max().item()

        # Update the Q-value for the chosen action
        self.optimizer.zero_grad()
        loss = self.loss_fn(q_values_old[action], torch.tensor(target_q_value))
        loss.backward()
        self.optimizer.step()

    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy policy.

        Returns:
            action: The chosen action.
        """

        state_tensor = torch.tensor(state, dtype=torch.float32)

        actions = ConnectFour.possible_actions(state)
        actions = list(actions)

        # Explore with probability epsilon
        if random.random() < self.epsilon:
            # Randomly choose an action
            action = random.choice(actions)
        else:
            # Choose action with maximum Q-value
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                #get only valid q_value
                valid_q_values = q_values[actions]

                if valid_q_values.numel() > 0:
                    #choose maximum valid q value
                    action = actions[torch.argmax(valid_q_values).item()]
                else:
                    #if no q value yet, choose random
                    action = random.choice(range(self.q_network.output_size))

        return action
       
    