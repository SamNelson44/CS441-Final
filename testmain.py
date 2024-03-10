from main import ConnectFour
import pytest

def test_placing_chips():
    game = ConnectFour()
    #chips size at column 1 should be 0
    assert game.chips_size[1] == 0
    game.place(1)
    #chips size at column 1 should be 1 now
    assert game.chips_size[1] == 1

def test_bad_column_inputs():
    game = ConnectFour()
    game.place(1)
    game.place(1)
    game.place(1)
    game.place(1)
    game.place(1)
    game.place(1)
    with pytest.raises(ValueError):
        game.place(1) #placing at 1 again should raise Value Error

