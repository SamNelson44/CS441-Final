import pygame
import sys
import numpy as np
import time
from minimax import MiniMax
import numpy as np

agent = MiniMax(max_depth=5, debug=False)
"""By default, X is 0, O is 1"""
# agent.state.board = [
#             [None, None, None, None, None, None, None],
#             [None, None, None, None, None, None, None],
#             [None, None, None, None, None, None, None],
#             [None, None, None, None, None, None, None],
#             ['O',  'O',  'O',   'X',  'X',  'X', None],
#             ['O',  'X',  'O',   'X',  'O',  'X', None]
#         ]
# agent.state.chips_size = [2,2,2,2,2,2,0]
agent.state.board = [[None]*7 for _ in range(6)]
agent.state.chips_size = [0]*7

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 700, 700
BOARD_ROWS, BOARD_COLS = 6, 7
SQUARE_SIZE = 100
RADIUS = SQUARE_SIZE // 2 - 5
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)

pygame.mixer.init()
pygame.mixer.music.load('ding.mp3')

# Function to create the game board
def create_board():
    return np.zeros((BOARD_ROWS, BOARD_COLS))

# Function to drop a piece onto the board
def drop_piece(board, row, col, piece):
    board[row][col] = piece


# def drop_animation(board, row, col, piece):
#     start_row = 0
#     while start_row < row:
#         win.fill((0, 0, 0))
#         draw_board(board)
#         pygame.draw.circle(win, (255, 255, 255), (col * SQUARE_SIZE + SQUARE_SIZE // 2, start_row * SQUARE_SIZE + SQUARE_SIZE + SQUARE_SIZE // 2), RADIUS)
#         pygame.display.update()
#         start_row += 1
#         time.sleep(0.1)

#     # Drop the piece onto the board
#     drop_piece(board, row, col, piece)

# Function to check if the selected column is valid for dropping a piece
def is_valid_location(board, col):
    return board[BOARD_ROWS - 1][col] == 0

# Function to get the next available row for dropping a piece in the selected column
def get_next_open_row(board, col):
    for r in range(BOARD_ROWS):
        if board[r][col] == 0:
            return r

# Function to check if there's a winning move
def winning_move(board, piece):
    # Check horizontal locations
    for c in range(BOARD_COLS - 3):
        for r in range(BOARD_ROWS):
            if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][c + 3] == piece:
                return True

    # Check vertical locations
    for c in range(BOARD_COLS):
        for r in range(BOARD_ROWS - 3):
            if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][c] == piece:
                return True

    # Check positively sloped diagonals
    for c in range(BOARD_COLS - 3):
        for r in range(BOARD_ROWS - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and board[r + 3][c + 3] == piece:
                return True

    # Check negatively sloped diagonals
    for c in range(BOARD_COLS - 3):
        for r in range(3, BOARD_ROWS):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and board[r - 3][c + 3] == piece:
                return True

# Initialize game variables
board = create_board()
turn = 0
game_over = False

# Set up the display
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Connect 4")

# Function to draw the game board
def draw_board(board):
    for c in range(BOARD_COLS):
        for r in range(BOARD_ROWS):
            pygame.draw.rect(win, BLUE, (c * SQUARE_SIZE, r * SQUARE_SIZE + SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            pygame.draw.circle(win, (0, 0, 0), (c * SQUARE_SIZE + SQUARE_SIZE // 2, r * SQUARE_SIZE + SQUARE_SIZE + SQUARE_SIZE // 2), RADIUS)

    for c in range(BOARD_COLS):
        for r in range(BOARD_ROWS):
            if board[r][c] == 1:
                pygame.draw.circle(win, RED, (c * SQUARE_SIZE + SQUARE_SIZE // 2, HEIGHT - r * SQUARE_SIZE - SQUARE_SIZE // 2), RADIUS)
            elif board[r][c] == 2:
                pygame.draw.circle(win, YELLOW, (c * SQUARE_SIZE + SQUARE_SIZE // 2, HEIGHT - r * SQUARE_SIZE - SQUARE_SIZE // 2), RADIUS)
    pygame.display.update()

# Function to display the win/lose/draw screen
def game_result(winner):
    font = pygame.font.Font(None, 36)
    if winner == 1:
        text = font.render("Red wins!", True, RED)
    elif winner == 2:
        text = font.render("Yellow wins!", True, YELLOW)
    else:
        text = font.render("It's a draw!", True, (255, 255, 255))
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    win.blit(text, text_rect)
    pygame.display.update()
    pygame.time.wait(3000)

def make_text(txt):
    font = pygame.font.Font(None, 36)
    text = font.render(txt, True, RED)
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    win.blit(text, text_rect)
    pygame.display.update()

# Function to restart the game
def restart_game():
    global board, turn, game_over
    board = create_board()
    turn = 0
    game_over = False
    win.fill((0, 0, 0))
    draw_board(board)

# Main game loop
while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        #EDIT THIS TO MAKE AI GO 1st or 2nd
        if turn == 1:

            make_text('AI is thinking. Please wait..')
            row, col = agent.mini_max_move(is_X=turn==0)
            # row,col =agent.random_move()
            agent.state.place(row, col)
            drop_piece(board, BOARD_ROWS-(row+1), col, 1)
            pygame.mixer.music.play()

            # Check for winning move
            if winning_move(board, 1):
                game_result(1)
                game_over = True
            turn += 1
            turn %= 2
            pygame.event.clear()
            

        if event.type == pygame.MOUSEBUTTONUP:

            pygame.draw.rect(win, (0, 0, 0), (0, 0, WIDTH, SQUARE_SIZE))
            # Get player input
            # if turn == 0:
                # posx = event.pos[0]
                # col = posx // SQUARE_SIZE

                # if is_valid_location(board, col):
                #     row = get_next_open_row(board, col)
                #     drop_piece(board, row, col, 1)
                #     # drop_animation(board, row, col, 1)

                
                


            # else:
            posx = event.pos[0]
            col = posx // SQUARE_SIZE

            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, 2)
                
                print('Next', row, col)
                agent.state.place(BOARD_ROWS-(row+1), col)
                # drop_animation(board, row, col, 2)

                if winning_move(board, 2):
                    game_result(2)
                    game_over = True

                # Draw the board after the animation
                draw_board(board)

                # Switch turns
                turn += 1
                turn %= 2

    # Clear the screen
    win.fill((0, 0, 0))

    # Draw the game board
    draw_board(board)

    # Update the display
    pygame.display.update()

    # Restart game if requested
    if game_over:
        restart_game()


