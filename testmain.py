from main import ConnectFour
import pytest
import unittest

WIDTH = 7

def test_placing_chips():
    game = ConnectFour()
    #chips size at column 1 should be 0
    assert game.chips_size[1] == 0
    game.place(1)
    #chips size at column 1 should be 1 now
    assert game.chips_size[1] == 1

def test_bad_column_inputs():
    game = ConnectFour()
    #check that possible column moves are 0-6
    state = game.get_state()
    assert game.possible_actions(state).issuperset({0,1,2,3,4,5,6})

    game.place(1)
    game.place(1)
    game.place(1)
    game.place(1)
    game.place(1)
    game.place(1)
    with pytest.raises(ValueError):
        game.place(1) #placing at 1 again should raise Value Error
    
    state = game.get_state()
    assert 1 not in game.possible_actions(state)

def test_terminal_states():
    game = ConnectFour()
    game.place(1)
    game.place(2)
    game.place(1)
    game.place(2)
    game.place(1)
    game.place(2)
    assert game.terminal() == None
    game.place(1) #game should end since 4 verticals for 1
    assert game.terminal() == 'X'

class TestConnectFour(unittest.TestCase):
    def setUp(self):
        # Initialize ConnectFour object before each test
        self.game = ConnectFour()

    def test_vertical_win(self):
        # Create a vertical win scenario
        self.game.board = [
            ['X', 'O', 'X', 'O', 'X', 'O', 'X'],
            ['X', 'O', 'X', 'O', 'X', 'O', 'X'],
            ['X', 'O', 'X', 'O', 'X', 'O', 'X'],
            ['X', 'O', 'X', 'O', 'X', 'O', 'X'],
            ['O', 'X', 'O', 'X', 'O', 'X', 'O'],
            ['O', 'X', 'O', 'X', 'O', 'X', 'O']
        ]
        for i in range(WIDTH):
            self.game.chips_size[i] = 6

        winner = self.game.terminal()
        self.assertEqual(winner, 'X', "Vertical win not detected")

    def test_no_winner(self):
        # Create a scenario where there is no winner
        self.game.board = [
            ['X', 'O', 'X', 'O', 'X', 'O', 'X'],
            ['O', 'X', 'O', 'X', 'O', 'X', 'O'],
            ['X', 'O', 'X', 'O', 'X', 'O', 'X'],
            ['X', 'O', 'X', 'O', 'X', 'O', 'X'],
            ['O', 'X', 'O', 'X', 'O', 'X', 'O'],
            ['O', 'X', 'O', 'X', 'O', 'X', 'O']
        ]
        for i in range(WIDTH):
            self.game.chips_size[i] = 6

        winner = self.game.terminal()
        self.assertIsNone(winner, f'Unexpected winer: {winner}')


    def test_horizontal_win(self):
        # Create a scenario where there is a horizontal win
        self.game.board = [
            ['X', 'X', 'X', 'X', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'X', 'X', 'X', 'X'],
            ['X', 'X', 'X', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'X', 'X', 'X', 'X'],
            ['O', 'X', 'O', 'X', 'O', 'X', 'O'],
            ['O', 'O', 'O', 'O', 'O', 'X', 'O']
        ]
        for i in range(WIDTH):
            self.game.chips_size[i] = 6

        winner = self.game.terminal()
        self.assertEqual(winner, 'O', "Horizontal win not detected")
    
    def test_diagonal_win(self):
        # Create a scenario where there is a horizontal win
        self.game.board = [
            ['X', 'O', 'X', 'O', 'X', 'O', 'X'],
            ['O', 'X', 'O', 'X', 'O', 'X', 'O'],
            ['X', 'O', 'X', 'X', 'X', 'O', 'X'],
            ['X', 'O', 'X', 'O', 'X', 'O', 'X'],
            ['O', 'X', 'O', 'X', 'O', 'X', 'O'],
            ['X', 'X', 'O', 'X', 'O', 'X', 'O']
        ]
        for i in range(WIDTH):
            self.game.chips_size[i] = 6

        winner = self.game.terminal()
        self.assertEqual(winner, 'X', "Diagonal Win not detected at row 6")

        self.game.board = [
            ['X', 'O', 'X', 'O', 'X', 'O', 'X'],
            ['O', 'X', 'O', 'X', 'O', 'X', 'O'],
            ['X', 'O', 'X', 'X', 'X', 'O', 'X'],
            ['X', 'O', 'O', 'O', 'X', 'O', 'X'],
            ['O', 'X', 'O', 'O', 'O', 'X', 'O'],
            ['O', 'X', 'O', 'X', 'O', 'X', 'O']
        ]
        winner = self.game.terminal()
        self.assertEqual(winner, 'O', "Diagonal win not detected")

        self.game.board = [
            ['X', 'X', 'X', 'O', 'X', 'O', 'X'],
            ['O', 'O', 'O', 'X', 'O', 'X', 'O'],
            ['X', 'X', 'X', 'O', 'X', 'O', 'X'],
            ['X', 'O', 'X', 'O', 'X', 'X', 'X'],
            ['O', 'X', 'O', 'O', 'O', 'X', 'O'],
            ['O', 'X', 'X', 'X', 'O', 'X', 'O']
        ]
        winner = self.game.terminal()
        self.assertEqual(winner, 'X', "Diagonal win not detected at (1,3)")



if __name__ == '__main__':
    unittest.main()

