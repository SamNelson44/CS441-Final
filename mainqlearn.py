from main import ConnectFour, EMPTY

from qlearn import Agent

import random

def train(n):
    ai = Agent()
    game_record = []
    starting_epsilon = ai.epsilon
    decrease_rate = 0.10

    def evaluate_ai():
        ai_score = 0
        for _ in range(100):
            game = ConnectFour()
            state = game.get_state()
            while True:
                action = ai.choose_action(state)
                next, reward, done = game.place(action)
                
                if done:
                    ai_score += reward
                    break
                random_moves = ConnectFour.possible_actions(game.board)
                random_move = random.choice(list(random_moves))
                next, reward, done = game.place(random_move)
                if done:
                    ai_score += reward
                    break
                state = next
                
        return ai_score

    for i in range(n):
        #Check performance every 1/10 of n
        if i % (n/10) == 0:
            score = evaluate_ai()
            game_record.append(score)
            if ai.epsilon > 0:
                ai.epsilon -= starting_epsilon * decrease_rate
                ai.epsilon = max(0, ai.epsilon)
       
   
        game = ConnectFour()
        while True:
            state = game.get_state()
            action = ai.choose_action(state)
            new_state , reward, done = game.place(action)
           
           #Evaluate the move
            if done:
                ai.update(state, action,new_state, reward)
                break
            else:
                ai.update(state, action,new_state, 0)
      
            random_moves = ConnectFour.possible_actions(new_state)
            random_move = random.choice(list(random_moves))
            new_state, reward, done = game.place(random_move)
            #evaluate the last AI move
            if done:
                ai.update(state, action, new_state, reward)
                break
            else:
                ai.update(state, action, new_state, 0)

    print("Done training")
    print(game_record)
    print(f'Last evaluation: {evaluate_ai()}')

    return game_record


if __name__ == '__main__':
    train(20000)