import random
import numpy as np
import copy

class TTT:
    EMPTY = 0
    PLAYER_X = 1
    PLAYER_O = -1

    def __init__(self):
        """
        Initialize the board
        Keep track of player
        """
        self.board = [[self.EMPTY, self.EMPTY, self.EMPTY],
                      [self.EMPTY, self.EMPTY, self.EMPTY],
                      [self.EMPTY, self.EMPTY, self.EMPTY]]
        self.current_player = self.PLAYER_X
        self.game_winner = None


    def place(self, action):
        """
        Make a move (i,j) in a tic tac toe grid

        Returns resulting board from move
        """
        i, j = action
        if self.board[i][j] != self.EMPTY:
            raise ValueError("Invalid move")
        else:
            self.board[i][j] = self.current_player
            self.current_player = self.other_player()  # Update the current player
            winner = self.winner()
            if not self.terminal():
                self.game_winner = winner
                done = False
            else:
                done = True
            return self.get_state(), done



    @classmethod
    def is_board_empty(cls, board):
        """
        Check if the board is empty, i.e., all cells are 0.
        """
        for row in board:
            for cell in row:
                if cell != cls.EMPTY:
                    return False
        return True

    def other_player(self):
        if self.current_player == self.PLAYER_X:
            return self.PLAYER_O
        else:
            return self.PLAYER_X
   
    def terminal(self):
        """
        Given a board, check if there is a winner

        Returns: True if game over, false otherwise
        """
        # A game is finished if there is a winner
        # Or if the board is full
        winner = self.winner()
        if winner is not None:
            self.game_winner = winner
            return True

        for i in range(3):
            for j in range(3):
                if self.board[i][j] == self.EMPTY:
                    return False
        
        return True
    
    def get_reward(self):
        if self.terminal():
            if self.game_winner == 1:
                return 1
            elif self.game_winner == -1:
                return -1
            else:
                return 0
        else:
            return 0
    

    @classmethod
    def possible_actions(cls, board):
        """
        Given a state board, return all possible action (i,j) for player
        """
        # a set make sense since there won't be duplicates
        result = set()

        # Double for loop and look for empty spots which are valid moves
        for i, row in enumerate(board):
            for j, cell in enumerate(row):
                if cell == cls.EMPTY:
                    result.add((i, j))
        return result
    
    def get_state(self):
        return np.array(self.board)

    def winner(self):
        """
        Returns who ever wins the game, none if there is no winner yet
        """

        # Check column winning and diagonals
        for i in range(3):
            current = self.board[0][i]
            if current != self.EMPTY:
                if self.board[1][i] == current and self.board[2][i] == current:
                    return current
                
                if i == 0:
                    if self.board[1][1] == current and self.board[2][2] == current:
                        return current
                
                # Check second diagonal
                if i == 2:
                    if self.board[1][1] == current and self.board[2][0] == current:
                        return current
            
        # Check row inner
        for i in range(3):
            current = self.board[i][0]
            if current != self.EMPTY:
                if self.board[i][1] == current and self.board[i][2] == current:
                    return current
        return None
    
    def __str__(self):
        result = ''
        for row in self.board:
            for cell in row:
                if cell == self.EMPTY:
                    result += '_'
                elif cell == self.PLAYER_X:
                    result += 'X'
                elif cell == self.PLAYER_O:
                    result += 'O'
            result += '\n'
        return result
    
    def reset(self):
        """
        Reset the game to its initial state
        """
        self.board = [[self.EMPTY for _ in range(3)] for _ in range(3)]
        self.current_player = self.PLAYER_X
        self.game_winner = None
        return self.get_state()
