import random
from main import ConnectFour, EMPTY

discount = 0.99
alpha = 0.3

class Agent:
    def __init__(self):
        """
        A RL learning agent uses Q table to approximate best action for states
        """
        self.q = dict()
        self.epsilon = 0.8
    
    def update(self, old_state, action, new_state, reward):
        """
        Update a value of Q(s,a) based on the previous state, new state, 
        and reward recieved from doing the action
        """
        old = self.get_q_value(old_state,action)
        future_rewards = self.best_action(new_state)
        self.update_q_value(old_state, action, old, reward, future_rewards)
    
    def get_q_value(self, state, action):
        """
        Return Q(s,a)
        If not exist yet, should be 0
        """
        try:
            #convert state into a tuple
            #This allows for hashing the state,action to a value
            state_key = tuple(map(tuple, state)) 
            key = (state_key, action)
            return self.q[key] 
        except KeyError: #if key doesn't exist then qvalue is 0
            return 0
    
    def best_action(self, state):
        """
        Given a state get the maximum q-value we can get from all possible action
        """
        actions = ConnectFour.possible_actions(state)
        highest_q = 0
        state_key = tuple(map(tuple, state))
        for action in actions:
            try:
                #attempt to get q value
                q_value = self.q[(state_key, action)]
            except KeyError:
                #if it doesn't exist, initialize it
                self.q[(state_key), action] = 0
            else:
                if q_value > highest_q:
                    highest_q = self.q[(state_key, action)]
        return highest_q

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """
        Using RL formula to calculate the new Q(s,a) value
        """
        new_estimate = reward + discount * (future_rewards)
        new_q = old_q + alpha * (new_estimate - old_q)
        value = self.q[(tuple(map(tuple,state)), action)] = new_q
        # print(value)
    
    def choose_action(self, state):
        """
        Given a game state, return action to make (i,j)

        """
        actions = ConnectFour.possible_actions(state)
        if not actions:
            actions = ConnectFour.possible_actions(state)
        state_key = tuple(map(tuple,state))
        highest_q = 0
        number = random.random()
        if number < self.epsilon:
            return random.choice(list(actions))
        best_action = list(actions)[0]
        for action in actions:
            try:
                current = self.q[(state_key, action)]
            except KeyError:
                self.q[(state_key, action)] = 0
                current = self.q[(state_key, action)]
            if current > highest_q:
                best_action = action
                highest_q = current
        return best_action


