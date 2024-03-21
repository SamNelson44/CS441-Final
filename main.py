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
            row = (HEIGHT - 1) - self.chips_size[column]
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
        Check if the game has ended
        """
        #check for vertical wins

        for column in range(WIDTH):
                #start at the first chip at [5][0] index
                current = self.board[HEIGHT -1][column]
                
                connecting = 1
                #next chip to compare should start at row 4, Height - 2
                for i in range(2, HEIGHT + 1):
                    if current== None:
                        break
                    next_row = HEIGHT - i
                    next_chip = self.board[next_row][column]
                    if next_chip != EMPTY and current == next_chip:
                        connecting += 1
                        if connecting == WINCONDITION:
                            return current
                    else:
                        current = next_chip
                        connecting = 1
        
        #check for horizontal
        for i in range(HEIGHT):
            row = HEIGHT - 1 - i
            current = self.board[row][0]
            connecting = 1

            for j in range(WIDTH -1):
                if current == None:
                    continue
                next_column = j + 1
                next_chip = self.board[row][next_column]
                if next_chip != EMPTY and current == next_chip:
                    connecting += 1
                    if connecting == WINCONDITION:
                        return current
                else:
                    current = next_chip
                    connecting = 1
        

        #checking for diagonals going upward left to right: /
        for column in range(WIDTH):
            #start at the first chip [5][0]
            current = self.board[HEIGHT -1][column]
            
            connecting = 1
            #next row to compare should be 4th row index, Height -2
            #next column to compare should be 2nd column index
            for i in range(2, HEIGHT + 1):
                if current== None:
                    break
                next_row = HEIGHT - i
                next_column = column + (i-1)
                try:
                    next_chip = self.board[next_row][next_column] #adjust column to be + 1 of current column
                except IndexError:
                    break #go to next chip if out of bounds
                if next_chip != EMPTY and current == next_chip:
                    connecting += 1
                    if connecting == WINCONDITION:
                        return current
                else:
                    current = next_chip
                    connecting = 1
        
        #check for diagonal downward going left to right: \    
        #start at row 2: HEIGHT(6) - (WINCONDITION)4
        current_row = HEIGHT - WINCONDITION
        current_col = 0
        while current_row>= 0:
            current = self.board[current_row][0]
            next_row = current_row
            next_col = current_col
            connecting = 1
            while True:
                next_row += 1
                next_col += 1
                #may be out of bounds array
                try:
                    next_chip = self.board[next_row][next_col]
                except IndexError:
                    #go to next row if we go out of array
                    current_row -= 1
                    break
                if next_chip != EMPTY and next_chip == current:
                    connecting += 1
                    if connecting == WINCONDITION:
                        return current
                else:
                    current = next_chip
                    connecting = 1

        #check for each column
        #start at col 1 until 3 away from WIDTH (WIDTH - (WINCONDITION - 1))
        current_row = 0
        current_col = 1
        while current_col < (WIDTH - (WINCONDITION -1)):
            connecting = 1
            current = self.board[current_row][current_col]
            next_row = current_row
            next_col = current_col
            while True:
                next_row += 1
                next_col += 1
                #may be out of bounds array
                try:
                    next_chip = self.board[next_row][next_col]
                except IndexError:
                    #go to next colum if we go out of array
                    current_col += 1
                    break
                if next_chip != EMPTY and next_chip == current:
                    connecting += 1
                    if connecting == WINCONDITION:
                        return current
                else:
                    current = next_chip
                    connecting = 1
        return None

                            




    
game = ConnectFour()
game.place(1)
game.place(1)
game.place(1)
game.place(1)
game.place(1)
game.place(1)
#placing one more at column 1 will raise exception
# game.place(1)
game.place(2)
print(game)