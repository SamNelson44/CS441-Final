from main import ConnectFour, EMPTY

from qlearn import Agent

import random


def print_ending_game(board):
    """
    Given a state array, print the board version with X, O
    """
    result = ""
    for row in board:
        for cell in row:
            if cell == 0:
                result += "_ "
            else:
                if cell == 1:
                    result += "X"
                elif cell == -1:
                    result += "O"
                result += " "
        result += "\n"
    print(result)


def train(n):
    ai = Agent()
    game_record = []
    starting_epsilon = ai.epsilon
    decrease_rate = 0.10
    print("Training AI:")
    print(f"epsilon: {starting_epsilon}")
    print(f"epsilon decrease_rate: {decrease_rate} every 1/10 of iteration")
    print(f"reward discount: {ai.discount}")
    print(f"Learning rate: {ai.alpha}")

    def evaluate_ai():
        """
        Evaluate the ai against random on 100 games
        Returns:
            Number of games (won, lost, draw) as a tuple
        """
        ai_win = 0
        ai_lost = 0
        draw = 0
        for _ in range(100):
            game = ConnectFour()
            state = game.get_state()
            while True:
                action = ai.choose_action(state)
                next, reward, done = game.place(action)

                if done:
                    if reward == 1:
                        ai_win += 1
                    else:
                        draw += 1
                    # print_ending_game(next)
                    break
                random_moves = ConnectFour.possible_actions(game.board)
                random_move = random.choice(list(random_moves))
                next, reward, done = game.place(random_move)
                if done:
                    if reward == -1:
                        ai_lost += 1
                    else:
                        draw += 1
                    # print_ending_game(next)
                    break
                state = next

        return ai_win, ai_lost, draw

    for i in range(n):
        # Check performance every 1/10 of n
        if i % (n / 10) == 0:
            win, lost, draw = evaluate_ai()
            game_record.append((win, lost, draw))
            print(
                f"Episode {i}: Winrate: {win}%, Lostrate: {lost}%, Draw:{draw}%, epsilon: {ai.epsilon}"
            )
            if ai.epsilon > 0:
                ai.epsilon -= starting_epsilon * decrease_rate
                ai.epsilon = max(0, ai.epsilon)

        game = ConnectFour()
        while True:
            state = game.get_state()
            action = ai.choose_action(state)
            new_state, reward, done = game.place(action)

            # Evaluate the move
            if done:
                ai.update(state, action, new_state, reward)
                break
            else:
                ai.update(state, action, new_state, 0)

            random_moves = ConnectFour.possible_actions(new_state)
            random_move = random.choice(list(random_moves))
            new_state, reward, done = game.place(random_move)
            # evaluate the last AI move
            if done:
                ai.update(state, action, new_state, reward)
                break
            else:
                ai.update(state, action, new_state, 0)

    print("Done training")
    # print(game_record)
    win, lost, draw = evaluate_ai()
    print(f"After training")
    print(f"Winrate: {win}%, Lostrate: {lost}%, Draw:{draw}%, epsilon: {ai.epsilon}\n")
    game_record.append((win, lost, draw))


if __name__ == "__main__":
    train(40000)
