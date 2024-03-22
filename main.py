import copy
import numpy as np

X = 'X'
O = 'O'
EMPTY = None
WIDTH= 7
HEIGHT = 6
WINCONDITION = 4

def check_valid_state(board):
    """
    Check if the current game state is valid.

    Args:
        board: The game board represented as a 2D list.

    Raises:
        ValueError: If the state is invalid.
    """
    for col in range(WIDTH):
        if board[0][col] is not None:
            raise ValueError("Invalid state: Chip at the very top in column {}".format(col))

    for col in range(WIDTH):
        for row in range(1, HEIGHT):
            if board[row][col] is not None and board[row - 1][col] is None:
                raise ValueError("Invalid state: Chip without support at column {}, row {}".format(col, row))

class ConnectFour:
    def __init__(self, state=None):
        """
        Initialize the game state

        Args:
            optional 2D state
            If None, the board will all be empty
        """
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.EMPTY = None
       
            
        if state is None:
            self.board = [[self.EMPTY for _ in range(self.WIDTH)] for _ in range(self.HEIGHT)]
            # Keep track of how many chips in a column by counting how many chips present in each column
            self.chips_size = [0] * WIDTH
        else:
            try:
                check_valid_state(state)
            except ValueError as e:
                raise ValueError(str(e))
            self.board = [row[:] for row in state]
            self.chips_size = [sum(1 for row in self.board if row[col] is not None) for col in range(self.WIDTH)]


        #start with X player
        self.current_player = 'X'
    
    def switch_player(self):
        """
        Swap self.current_player
        """
        if self.current_player == 'X':
            self.current_player = 'O'
        else:
            self.current_player = 'X'
    
    def __str__(self):
        result = ""
        for row in self.board:
            for cell in row:
                if cell is EMPTY:
                    result += '_ '
                else:
                    result += cell + ' '
            result += '\n'
        return result
    
    @classmethod
    def is_board_empty(cls, board):
        """
        Check if all positions on the board are empty.

        Args:
            board: The game board represented as a 2D list.

        Returns:
            bool: True if all positions are empty, False otherwise.
        """
        for row in board:
            for cell in row:
                if cell is not EMPTY:
                    return False
        return True
    
    def place(self, column):
        """
        Place a chip for self.current_player and update the game state.

        Parameters:
            column: Which column to place chip

        Returns:
            next_state : State after actioin
            reward: The reward obtained by the agent for the action.
            done (bool): A flag indicating whether the episode has terminated.
        """
        if column < 0 or column >= self.WIDTH:
            raise ValueError(f'Invalid column: {column}')

        if self.chips_size[column] == self.HEIGHT:
            raise ValueError(f'Too many pieces in column {column}')

        # Find the next available row to place the chip
        row = self.chips_size[column]
        self.board[row][column] = self.current_player
        self.chips_size[column] += 1

        # Check if the current player has won
        winner = self.terminal()
        if winner is not None:
            done = True
            # Give a reward if agent won, -1 if lost
            reward = 1.0 if winner == 'X' else -1.0

        # Check if the game board is full (draw)
        elif all(self.board[row][col] is not None for col in range(self.WIDTH) for row in range(self.HEIGHT)):
            done = True
            reward = 0.5
        else:
            #game hasn't ended so no rewards
            done = False
            reward = 0.0

        # Switch the current player for the next turn
        self.switch_player()

        # Return the updated state, reward, and whether game ends
        next_state = self.get_state()
        return next_state, reward, done
    
    def get_state(self):
        """
        Returns: A numerical 2D list X: 1, O: -1, None : 0
            This is so that tensors can use the state
        """
        state_array = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.int32)
        for i in range(self.HEIGHT):
            for j in range(self.WIDTH):
                if self.board[i][j] == 'X':
                    state_array[i][j] = 1  # Player 'X' represented by 1
                elif self.board[i][j] == 'O':
                    state_array[i][j] = -1  # Player 'O' represented by -1

        return copy.deepcopy(state_array)
    
    
    @classmethod
    def possible_actions(cls, state):
        """
        Given a game state 2D array, calculate possible place to drop column

        Returns a set of numbers of possible action
        """
        actions = set()

        #if there is an empty spot at the top, we can drop chips at the column
        for column in range(WIDTH):
            # if state[HEIGHT -1][column] == None and state[HEIGHT -1][column] == 0:
            if state[HEIGHT -1][column] == 0 or state[HEIGHT -1][column] == None:
                actions.add(column)
        return actions

    def terminal(self):
        """
        Check if the game has a winner or draw

        Returns:
            'X' or 'O' if there is one winner.  None if no winner
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for row in range(HEIGHT):
            for col in range(WIDTH):
                for dy, dx in directions:
                    winner = self.check_consecutive(row, col, dy, dx)
                    if winner is not None:
                        return winner
        return None

    def check_consecutive(self, row, col, dy, dx):
        """
        Parameters:
            row, col: starting location for chip
            dy, dx : which way to compare chips
        Returns:
            'X', 'O', or None if no winner
        """
        current_chip = self.board[row][col]
        if current_chip is None:
            return None

        for _ in range(WINCONDITION - 1):
            row += dy
            col += dx
            # Check if position is out of bounds
            if not (0 <= row < HEIGHT and 0 <= col < WIDTH):
                return None
            if self.board[row][col] != current_chip:
                return None
        return current_chip
    
    def reset(self):
        self.board = [[self.EMPTY for _ in range(self.WIDTH)] for _ in range(self.HEIGHT)]
        self.chips_size = [0] * WIDTH
        return self.get_state()
        

