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
    
    def inverted_board(self):
        tmp = deepcopy(self.board)
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if tmp[i][j] and tmp[i][j] == 'X':
                    tmp[i][j] = 'O'
                elif tmp[i][j] and tmp[i][j] == 'O':
                    tmp[i][j] = 'X'
        return tmp

class MiniMax():
    def __init__(self, discount = 1/7, max_depth = 2, debug = False) -> None:
        """max_depth: works best with even number!"""
        self.state = State()
        self.debug = debug
        self.MAX_DEPTH = max_depth
        self.discount = discount
        self.depth_discount = 0.95
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
                if is_x:
                    tot += self.attack * v * pow(self.discount, 4 - min(4, k)) * pow(self.depth_discount, depth)
                else:
                    tot += self.defense * v * pow(self.discount, 4 - min(4, k)) * pow(self.depth_discount, depth)
        return tot

    def mini_max_helper(self, state, depth, get_max: bool):
        random.seed(int(time.time()))
        def get_best(movs, evals, get_max):
            if get_max:
                return sorted(list(zip(movs, evals)), reverse=True, key=lambda x:x[1])[0]
            return sorted(list(zip(movs, evals)), key=lambda x:x[1])[0]
        
        nxt_moves = self.state.next_possible_moves()
        #opponent move
        if not get_max:
            board = state.inverted_board()
        else:
            board = state.board
        eval_movs = []
        for move in nxt_moves:
            #drill down (DFS)
            if depth < self.MAX_DEPTH:
                nxt_state = deepcopy(state)

                # nxt_state.place(move[1])
                nxt_state.board[move[0]][move[1]] = nxt_state.current_player
                nxt_state.chips_size[move[1]] += 1
                nxt_state.switch_player()

                nxt_mov, nxt_score = self.mini_max_helper(nxt_state, depth+1, not get_max)
                record_x = self.count_lines(board, move[0], move[1])
                record_o = self.count_lines(state.inverted_board(), move[0], move[1])
                #best score can get from subsequent moves + current board reward
                eval_movs += [self.evaluate_move(record_x, True, depth) + self.evaluate_move(record_o, False, depth) + nxt_score]
            # leaf node
            else:
                record_x = self.count_lines(board, move[0], move[1])
                record_o = self.count_lines(state.inverted_board(), move[0], move[1])
                eval_movs += [self.evaluate_move(record_x, True, depth) + self.evaluate_move(record_o, False, depth)]
        
        if depth == self.MAX_DEPTH:
            mov, score = get_best(nxt_moves, eval_movs, get_max=True)
        else:
            mov, score = get_best(nxt_moves, eval_movs, get_max)
        return (mov, score)
    
    def mini_max_move(self):
        """CURRENTLY ONLY APPLICABLE FOR X PLAYER"""
        best_mov , best_score = self.mini_max_helper(self.state, 0, True)
        self.print(f"Best score: {best_score:.5f} for {best_mov}")
        # return best_mov[1]
        return best_mov
    
    def random_move(self):
        nxt_moves = self.state.next_possible_moves()
        choice = np.random.choice(range(len(nxt_moves)), 1)[0]
        #return only the column
        # return nxt_moves[choice][1]
        return nxt_moves[choice]

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