from main import ConnectFour, WIDTH, HEIGHT
from collections import defaultdict
from copy import deepcopy
import numpy as np
import random
import time

class State(ConnectFour):
    def __init__(self):
        super().__init__()

    def next_possible_moves(self):
        res = []
        for idx,chips in enumerate(self.chips_size):
            if chips < HEIGHT:
                res += [(HEIGHT-chips-1, idx)]
        return res
    
    def place(self, row, col):
        if self.board[row][col] == 'X' or self.board[row][col] == 'O':
            raise ValueError(f'Board position is already taken at {row},{col}?\n {self.board} \n{self.chips_size}\n{self.next_possible_moves()}')
        self.board[row][col] = self.current_player

        #increase chip size for the column
        self.chips_size[col] += 1
        self.switch_player()
    
    def inverted_board(self):
        tmp = deepcopy(self.board)
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if tmp[i][j] and tmp[i][j] == 'X':
                    tmp[i][j] = 'O'
                elif tmp[i][j] and tmp[i][j] == 'O':
                    tmp[i][j] = 'X'
        return tmp
    
    def reset_board(self):
        self.board = [[None]*7 for _ in range(6)]
        self.chips_size = [0]*7

class MiniMax():
    def __init__(self, discount = 1/7, max_depth = 2, debug = False) -> None:
        """max_depth: works best with even number!"""
        self.state = State()
        self.debug = debug
        self.MAX_DEPTH = max_depth
        self.discount = discount
        self.depth_discount = 1/7
        self.attack = 1
        self.defense = 0.9

    def print(self, *args):
        if self.debug:
            print(*args)

    def evaluate_move(self, record: dict, is_x: bool, depth) -> float:
        tot = 0
        for k,v in record.items():
            #only accounts for 2 or more connected dots
            if k >= 2:
                # 4+ connected dots are calculated the same as 4
                # base_score * count * discount_connected * discount_depth 
                if is_x:
                    tot += self.attack * pow(self.discount, 4 - min(4, k)) * pow(self.depth_discount, depth)
                else:
                    tot += self.defense * pow(self.discount, 4 - min(4, k)) * pow(self.depth_discount, depth)
        return tot

    def mini_max_helper(self, state, depth, get_max: bool):
        def get_best(movs, evals, get_max):
            if len(movs) == 0:
                return (None, 0)
            #size >= 3 and no zero
            # test_eval = list(map(lambda x: x == 0, sorted(evals, reverse=get_max)[:3]))
            # if len(test_eval) > 2 and not any(test_eval):
            #     #normalized choice
            #     s = sum(sorted(evals, reverse=get_max)[:3])
            #     p = list(map(lambda x: x/s, sorted(evals, reverse=get_max)[:3]))
            #     top3 = np.random.choice(range(3), size=1, p= p)[0]
            # else:
            #     top3 = 0
            top3 = 0
            best_mov, best_score = sorted(list(zip(movs, evals)), reverse=True, key=lambda x:x[1])[top3]
            if get_max:
                return (best_mov, best_score)
            return (best_mov, best_score*-1)
        
        nxt_moves = state.next_possible_moves()
        #opponent move
        if not get_max:
            board = state.inverted_board()
        else:
            board = state.board
        eval_movs = []
        for move in nxt_moves:
            #drill down (DFS)
            if depth < self.MAX_DEPTH:

                record_x = self.count_lines(board, move[0], move[1])
                record_o = self.count_lines(state.inverted_board(), move[0], move[1])

                #winning move -> do not need to look further down.
                if 4 in record_x.keys():
                    eval_movs += [self.evaluate_move(record_x, True, depth) + self.evaluate_move(record_o, False, depth)]
                else:
                    nxt_state = deepcopy(state)
                    nxt_state.place(move[0], move[1])
                    nxt_mov, nxt_score = self.mini_max_helper(nxt_state, depth+1, not get_max)
                    #best score can get from subsequent moves + current board reward
                    eval_movs += [self.evaluate_move(record_x, True, depth) + self.evaluate_move(record_o, False, depth) + nxt_score]

                # nxt_state.board[move[0]][move[1]] = nxt_state.current_player
                # nxt_state.chips_size[move[1]] += 1
                # nxt_state.switch_player()                
                
            # leaf node
            else:
                record_x = self.count_lines(board, move[0], move[1])
                record_o = self.count_lines(state.inverted_board(), move[0], move[1])
                eval_movs += [self.evaluate_move(record_x, True, depth) + self.evaluate_move(record_o, False, depth)]
        
        mov, score = get_best(nxt_moves, eval_movs, get_max)
        return (mov, score)
    
    def mini_max_move(self, is_X = True):
        """CURRENTLY ONLY APPLICABLE FOR X PLAYER"""
        if is_X:
            best_mov , best_score = self.mini_max_helper(self.state, 0, True)
        else:
            inverted_state = deepcopy(self.state)
            inverted_state.board = inverted_state.inverted_board()
            inverted_state.switch_player()
            best_mov , best_score = self.mini_max_helper(self.state, 0, True)
        self.print(f"Best score for {self.state.current_player}: {best_score:.5f} for {best_mov}")
        # return best_mov[1]
        return best_mov
    
    def random_move(self):
        nxt_moves = self.state.next_possible_moves()
        choice = np.random.choice(range(len(nxt_moves)), 1)[0]
        #return only the column
        # return nxt_moves[choice][1]
        return nxt_moves[choice]        

    def check_status(self, i, j):
        def win(i, j):
            if self.state.current_player == 'X':
                record = self.count_lines(self.state.board, i, j)
            else:
                record = self.count_lines(self.state.inverted_board(), i, j)
            return 4 in record.keys()
        
        if win(i, j):
            if self.state.current_player == 'X':
                return 'X wins'
            else:
                return 'O wins'
        #if there is only one move left -> game over
        if len(self.state.next_possible_moves()) == 1:
            return 'Draw!'
        return None

    def count_lines(self, board, i, j):
        def max_continous(arr):
            max_continuous_zeros = 0
            current_continuous_zeros = 0
            
            for num in arr:
                if num and num == 'X':
                    current_continuous_zeros += 1
                elif num and num == 'O':
                    return 0
                else:
                    max_continuous_zeros = max(max_continuous_zeros, current_continuous_zeros)
            
            return max(max_continuous_zeros, current_continuous_zeros)
        
        def valid_coor(x,y):
            return x >= 0 and x < HEIGHT and y >= 0 and y < WIDTH               
        connected = defaultdict(int)
        board[i][j] = 'X'

        # Horizontal
        tmp = [0]
        for a in range(max(0,j-3), min(j+4,WIDTH)):
            b = min(a + 3, WIDTH-1)
            if b - a + 1 == 4:
                # self.print([(i,x) for x in range(a,b+1)])
                # self.print([board[i][x] for x in range(a,b+1)])
                tmp += [max_continous([board[i][x] for x in range(a,b+1)])]
        connected[max(tmp)] += 1
            
        # Vertical
        tmp = [0]
        for a in range(max(0,i-3), min(i+4,HEIGHT)):
            b = min(a + 3, HEIGHT-1)
            if b - a + 1 == 4:
                # self.print([(x,j) for x in range(a,b+1)])
                # self.print([board[x][j] for x in range(a,b+1)])
                tmp += [max_continous([board[x][j] for x in range(a,b+1)])]
        connected[max(tmp)] += 1

        # Diagonal (top-left to bottom-right)
        tmp = [0]
        for c in range(-3, 4):
            m = i + c; n = j + c
            u = m + 3; v = n + 3
            if valid_coor(m,n) and valid_coor(u,v):
                # self.print([(m+d,n+d) for d in range(0, 4)])
                # self.print([board[m+d][n+d] for d in range(0, 4)])
                tmp += [max_continous([board[m+d][n+d] for d in range(0, 4)])]
        connected[max(tmp)] += 1

        # Diagonal (bottom-left to top-right)
        tmp = [0]
        for c in range(-3, 4):
            m = i - c; n = j + c
            u = m - 3; v = n + 3
            if valid_coor(m,n) and valid_coor(u,v):
                # self.print([(m-d,n+d) for d in range(0, 4)])
                # self.print([board[m-d][n+d] for d in range(0, 4)])
                tmp += [max_continous([board[m-d][n+d] for d in range(0, 4)])]
        connected[max(tmp)] += 1

        board[i][j] = None
        return connected