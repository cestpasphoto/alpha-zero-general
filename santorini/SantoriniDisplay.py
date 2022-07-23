import numpy as np
from colorama import Style, Fore, Back
import random
import itertools

my_workers_color    = [Fore.WHITE, Fore.BLUE  , Fore.CYAN]
other_workers_color = [Fore.WHITE, Fore.YELLOW, Fore.MAGENTA]
# levels_char = ['▪', '◔', '◑', '◕', 'X']
levels_char = ['◎', '▂', '▅', '█', 'X']
directions_char = ['↖', '↑', '↗', '←', '→', '↙', '↓', '↘']


def move_to_str(move, player):
	# Decode move
	worker, move_ = divmod(move, 8*8)
	move_direction, build_direction = divmod(move_, 8)
	worker_color = my_workers_color[worker+1] if player == 0 else other_workers_color[worker+1]

	return f'Move {worker_color}worker {worker+1}{Fore.WHITE} to {directions_char[move_direction]} and then build {directions_char[build_direction]}'


############################# PRINT GAME ######################################

def _print_colors():
	print(f'Player 0: {my_workers_color[1]}worker 1  {my_workers_color[2]}worker 2    {Fore.WHITE}Player 1: {other_workers_color[1]}worker 1  {other_workers_color[2]}worker 2{Fore.WHITE}')

def _print_main(board):
	print(f'-'*11)
	for y in range(5):
		for x in range(5):
			worker, level = board.state[y, x, :]
			worker_color = my_workers_color[worker] if worker >= 0 else other_workers_color[-worker]
			if worker != 0 or level > 0:
				print(f'|{worker_color}{levels_char[level]}{Fore.WHITE}', end='')
			else:
				print(f'| ', end='')
		print('|')
		print(f'-'*11)

def print_board(board):
	print()
	# _print_round_and_scores(board)
	_print_colors()
	_print_main(board)