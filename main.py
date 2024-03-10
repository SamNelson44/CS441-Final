
X = 'X'
O = 'O'
EMPTY = None
WIDTH= 7
HEIGHT = 6

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