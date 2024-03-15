import copy

X = 'X'
O = 'O'
EMPTY = None
WIDTH= 7
HEIGHT = 6
WINCONDITION = 4

class ConnectFour:
    def __init__(self):
        """
        Initialize the game state
        """
        self.board = [[EMPTY for _ in range(WIDTH)] for _ in range(HEIGHT)]

        #keep track of how many chips in a column
        #initially [0,0,0,0,0,0,0] if WIDTH is 7
        #[0,1,0,0,0,0,0] means 1 chip at column 2
        self.chips_size = [0] * WIDTH

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
    
    def place(self, column):
        """
        place a chip for self.current_player
        """
        if column < 0 or column >= WIDTH:
            raise ValueError(f'Invalid column :{column}')
        elif self.chips_size[column] == HEIGHT:
            raise ValueError(f'Too many pieces at column {column}')
        else:
            #check which row to add the chip using chips_size at column
            #add at the end of the array
            row = self.chips_size[column]
            self.board[row][column] = self.current_player

            #increase chip size for the column
            self.chips_size[column] += 1
            self.switch_player()
    
    def get_state(self):
        """
        Returns: A copy of current state representation using deep copy
        """
        return copy.deepcopy(self.board)
    
    @classmethod
    def possible_actions(cls, state):
        """
        Given a game state 2D array, calculate possible place to drop column

        Returns a set of numbers of possible action
        """
        actions = set()

        #if there is an empty spot at the top, we can drop chips at the column
        for column in range(WIDTH):
            if state[0][column] == EMPTY:
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

# game = ConnectFour()
# game.place(1)
# game.place(1)
# game.place(1)
# game.place(1)
# game.place(1)
# game.place(1)
#placing one more at column 1 will raise exception
# game.place(1)
# game.place(2)
# print(game)