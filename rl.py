from main import ConnectFour, WIDTH, HEIGHT
from collections import defaultdict
from copy import deepcopy
import numpy as np
import random
import time
from random import shuffle
from collections import defaultdict
import re

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

    def make_move(self, row, col):
        if self.board[row][col] == 'X' or self.board[row][col] == 'O':
            raise ValueError(f'Board position is already taken at {row},{col}?\n {self.board} \n{self.chips_size}\n{self.next_possible_moves()}')
        nxt = State()
        nxt.current_player = self.current_player
        nxt.board = deepcopy(self.board)
        nxt.board[row][col] = self.current_player

        #increase chip size for the column
        nxt.chips_size = deepcopy(self.chips_size)
        nxt.chips_size[col] += 1
        nxt.switch_player()
        return nxt
    
    def inverted_board(self):
        tmp = deepcopy(self.board)
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if tmp[i][j] and tmp[i][j] == 'X':
                    tmp[i][j] = 'O'
                elif tmp[i][j] and tmp[i][j] == 'O':
                    tmp[i][j] = 'X'
        return tmp
    
    def inverted_state(self):
        self.board = self.inverted_board()
        if self.current_player == 'X':
            self.current_player = 'O'
        else:
            self.current_player = 'X' 
    
    def reset_board(self):
        self.board = [[None]*7 for _ in range(6)]
        self.chips_size = [0]*7

    def board_to_string(self):
        s = ''
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j] and self.board[i][j] == 'X':
                    s += '1'
                elif self.board[i][j] and self.board[i][j] == 'O':
                    s += '2'
                else:
                    s += '0'
        return s
    
    def __hash__(self):
        return int(self.board_to_string)
    
    def __str__(self):
        return self.board_to_string()
    


class RL():
    def __init__(self, eps=0.8, delta=0.01, m_ep=1, min_eps= 0.05, epochs=100, lr= 0.001, gamma=0.9) -> None:
        self.state = State()
        """Initialize the game object where:
        eps [0,1]      : epsilon-greedy percentage -> optimized move
        delta [0,1]    : decaying epsilon after M iterations
        m_ep [1:inf]   : number of iterations when delta decay occurs
        min_eps [0,1]  : minimum epsilon after decay
        epochs [1: inf]: number of epochs to self-play
        lr [0,1]       : learning rate for when updating Q(s, a)
        gamma [0,1]    : spreading factor for the next optimal move Q(s,a)"""
        self.eps=eps
        self.delta=delta
        self.m_ep=m_ep
        self.epochs=int(epochs)
        self.min_eps = min_eps
        self.lr=lr
        self.gamma = gamma

    def print(self, *args):
        if self.debug:
            print(*args)

    def self_play(self, start_state: State):
        state = start_state
        end = False
        while len(state.next_possible_moves()) > 0 and not end:
            #get possible moves
            movs = state.next_possible_moves()

            re_next_mov = []
            if state.current_player == 'O':
                state.inverted_state()
            
            #evaluate next state after each move
            for mov in movs:
                val = self.check_board_status(state, mov[0], mov[1])
                nxt_state = state.make_move(mov[0], mov[1])
                #state + col
                enc = str(state) + str(mov[1])

                re_next_mov += [(nxt_state, mov, val)]

            _, _, best_re = sorted(re_next_mov, key=lambda x:x[2], reverse=True)[0]
            #none of the next state is a winning state
            if best_re != 1:
                for mov in movs:
                    nxt_state = state.make_move(mov[0], mov[1])
                    op_movs = nxt_state.next_possible_moves()
                    for op_mov in op_movs:
                        val = self.check_board_status(nxt_state, op_mov[0], op_mov[1])
                        # oppponent winning move -> take defensive move is encourage
                        if val == 1:
                            re_next_mov += [(state, op_mov, 0.9)]
                        # return the config where it is our next move evaluation state assuming opponent take move x
                        else:
                            op_nxt_state = nxt_state.make_move(op_mov[0], op_mov[1])
                            #possible moves from ourselves
                            our_movs = op_nxt_state.next_possible_moves()
                            future_re_lst = []
                            for our_mov in our_movs:
                                future_re_lst += [self.check_board_status(op_nxt_state, our_mov[0], our_mov[1])]
                            if len(future_re_lst) != 0:
                                # re_next_mov += [(state, mov, sum(future_re_lst)/len(future_re_lst))]
                                re_next_mov += [(state, mov, max(future_re_lst))]

            #steps that have score that are insignificant difference should have equal chance at making
            # re_next_mov = list(map(lambda x: (x[0],x[1],round(x[2],2)), re_next_mov))
            _, _, best_re = sorted(re_next_mov, key=lambda x:x[2], reverse=True)[0]
            re_best_mov = list(filter(lambda x: x[2] == best_re, re_next_mov))
            
            nxt_state = None
            #eps-greedy best move is made
            if not np.random.choice([True, False], 1, p=[self.eps, 1 - self.eps])[0] and state.current_player == 'X':
                _, last_mov, nxt_r = re_best_mov[np.random.choice(len(re_best_mov),1)[0]]
                # _, last_mov, nxt_r = re_best_mov[0]
                if self.check_board_status(state, last_mov[0], last_mov[1]) != 0:
                    end = True
                # state + col
                enc = str(state) + str(last_mov[1])
                nxt_state = state.make_move(last_mov[0], last_mov[1])



                self.q_matrix[enc] = self.stable_q_matrix[enc] + self.lr*(nxt_r + self.gamma*best_re - self.stable_q_matrix[enc])
            else:
                _, last_mov, nxt_r = re_next_mov[np.random.choice(len(re_next_mov),1)[0]]
                # _, last_mov, nxt_r = re_next_mov[0]
                if self.check_board_status(state, last_mov[0], last_mov[1]) != 0:
                    end = True

                enc = str(state) + str(last_mov[1])
                nxt_state = state.make_move(last_mov[0],last_mov[1])

                self.q_matrix[enc] = self.stable_q_matrix[enc] + self.lr*(nxt_r + self.gamma*best_re - self.stable_q_matrix[enc])

            state = nxt_state

    def train(self):
        if not hasattr(self, 'stable_q_matrix'):
            self.stable_q_matrix = defaultdict(float)
        if not hasattr(self, 'q_matrix'):
            self.q_matrix = defaultdict(float)
        if not hasattr(self, 'history'):
            self.history = []
        cnt = 1
        for i in range(self.epochs):
            print(f"Epoch {i}/{self.epochs}  -  {i/self.epochs*100:.2f}%", end='\r')
            #X goes first
            x_state = State()
            x_state.current_player = 'X'
            self.self_play(x_state)
            #O goes first
            o_state = State()
            o_state.current_player = 'O'
            self.self_play(o_state)

            if i % 100 == 0:
                self.stable_q_matrix.update(self.q_matrix)

            if i % 10 == 0:
                tot = 0
                for _ in range(5):
                    for turn in ['X','O']:
                        state = State()
                        state.current_player = turn
                        winner = self.vs_random(state)
                        if winner == turn:
                            tot += 1
                        elif winner == '-':
                            tot += 0.5
                
                self.history += [(tot,'')]

            #update new epsilon
            if cnt % self.m_ep == 0:
                self.eps = max(self.min_eps, self.eps - self.delta)
            cnt += 1

    def vs_random(self, start_state: State, regr = None):
        state = start_state
        end = False
        while len(state.next_possible_moves()) > 0 and not end:
            #get possible moves
            movs = state.next_possible_moves()

            re_next_mov = []
            #evaluate next state after each move
            if regr:
                def convert(arr):
                    return list(map(lambda x:-1 if x == '2' else int(x), arr))
                feat = np.array(list(map(lambda x:convert(x), list(str(state))))).reshape(1, -1)
                q_vals = regr.predict(feat).tolist()[0]
            for mov in movs:
                enc = str(state) + str(mov[1])
                if self.stable_q_matrix[enc] == 0 and regr:
                    re_next_mov += [(state, mov, q_vals[mov[1]])]
                else:
                    re_next_mov += [(state, mov, self.stable_q_matrix[enc])]

            best_re = sorted(re_next_mov, key=lambda x:x[2], reverse=True)[0][2]
            re_best_mov = list(filter(lambda x: x[2] == best_re, re_next_mov))

            nxt_state = None
            last_mov = None
            #eps-greedy best move is made
            if state.current_player == 'X':
                last_mov = re_best_mov[np.random.choice(len(re_best_mov),1)[0]][1]
                if self.check_board_status(state, last_mov[0], last_mov[1]) != 0:
                    end = True
                nxt_state = state.make_move(last_mov[0], last_mov[1])
            else:
                last_mov = np.random.choice(range(len(movs)),1)[0]
                last_mov = movs[last_mov]
                if self.check_board_status(state, last_mov[0], last_mov[1]) != 0:
                    end = True
                nxt_state = state.make_move(last_mov[0],last_mov[1])

            state = nxt_state

        if len(state.next_possible_moves()) == 0:
            return '-'
        return state.current_player
    
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
        if sum(self.state.chips_size) == 41:
            # print(self.state.chips_size)
            return 'Draw!'
        return None
    
    def check_board_status(self, state, i, j):
        def win(i, j):
            if state.current_player == 'X':
                record = self.count_lines(state.board, i, j)
            else:
                record = self.count_lines(state.inverted_board(), i, j)
            return 4 in record.keys()
        
        if win(i, j):
            return 1
        #if there is only one move left -> game over
        if sum(state.chips_size) == 41:
            # print(self.state.chips_size)
            return 1e-4
        return 0

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