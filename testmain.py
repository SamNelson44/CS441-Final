from main import ConnectFour
import pytest
import unittest

WIDTH = 7


class TestConnectFour(unittest.TestCase):
    def setUp(self):
        # Initialize ConnectFour object before each test
        self.game = ConnectFour()

    # Test placing chips and chips_size increments
    def test_placing_chips(self):
        self.assertEqual(self.game.chips_size[1], 0)  
        self.game.place(1)
        self.assertEqual(self.game.chips_size[1], 1)  

    # Test bad column inputs
    def test_bad_column_inputs(self):
        state = self.game.get_state()
        # check that possible column moves are 0-6
        self.assertTrue(self.game.possible_actions(state).issuperset({0, 1, 2, 3, 4, 5, 6})) 

        for _ in range(6):
            self.game.place(1)

        with self.assertRaises(ValueError):
            self.game.place(1)  # placing at 1 again should raise Value Error

        state = self.game.get_state()
        self.assertNotIn(1, self.game.possible_actions(state))

    def test_terminal_states(self):
        # Test terminal states
        self.game.place(1)
        self.game.place(2)
        self.game.place(1)
        self.game.place(2)
        self.game.place(1)
        self.game.place(2)
        self.assertIsNone(self.game.terminal())  # game should not end yet
        self.game.place(1)  # game should end since 4 verticals for 1
        self.assertEqual(self.game.terminal(), 'X')

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
            ['X', 'O', 'O', 'O', 'O', 'O', 'X'],
            ['O', 'X', 'O', 'X', 'O', 'X', 'O'],
            ['X', 'O', 'X', 'O', 'X', 'O', 'X'],
            ['X', 'O', 'X', 'O', 'X', 'O', 'X'],
            ['O', 'X', 'O', 'X', 'O', 'X', 'O'],
            ['O', 'X', 'O', 'X', 'O', 'X', 'O']
        ]
        for i in range(WIDTH):
            self.game.chips_size[i] = 6

        winner = self.game.terminal()
        self.assertEqual(winner, 'O', "Horizontal O win not detected")


        self.game.board = [
            ['O', None,'O','X','X','X','X'],
            [None, None, None, None,'O', None, None],
            [None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None],
        ]

        self.game.chips_size = [1,0,1,1,2,1,1]
        winner = self.game.terminal()
        self.assertEqual(winner, 'X', "Horizontal X win not detected")

        self.game.board = [
            ['X', 'X', 'X', 'X', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'X', 'X', 'X', 'X'],
            ['X', 'X', 'X', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'X', 'X', 'X', 'X'],
            ['O', 'X', 'O', 'O', 'O', 'X', 'O'],
            ['X', 'O', 'X', 'O', 'O', 'O', 'O']
        ]
        for i in range(WIDTH):
            self.game.chips_size[i] = 6

        winner = self.game.terminal()
        self.assertEqual(winner, 'X', "Horizontal win not detected")

        self.game.board = [
            ['O', None,'O','X','X','X','X'],
            [None, None, None, None,'O', None, None],
            [None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None],
        ]
        winner = self.game.terminal()
        self.assertEqual(winner, 'X', "Horizontal win not detected")
    
    def test_diagonal_win(self):
        self.game.board = [
            ['X', 'O', 'X', 'O', 'X', 'O', 'X'],
            ['O', 'X', 'O', 'X', 'O', 'X', 'O'],
            ['X', 'O', 'X', 'X', 'X', 'O', 'X'],
            ['X', 'O', 'X', 'O', 'X', 'O', 'X'],
            ['O', 'X', 'O', 'X', 'O', 'X', 'O'],
            ['X', 'X', 'O', 'X', 'O', 'X', 'O']
        ]
   
        winner = self.game.terminal()
        self.assertEqual(winner, 'X', "Diagonal X Win not detected at row 6")

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

