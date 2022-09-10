import numpy as np
from colorama import Style, Fore, Back
import random
import itertools
from .SantoriniConstants import _decode_action, NB_GODS, NO_GOD

my_workers_color    = [Fore.WHITE, Fore.BLUE  , Fore.CYAN]
other_workers_color = [Fore.WHITE, Fore.YELLOW, Fore.MAGENTA]
# levels_char = ['▪', '◔', '◑', '◕', 'X']
levels_char = ['◎', '▂', '▅', '█', 'X']
directions_char = ['↖', '↑', '↗', '←', 'Ø', '→', '↙', '↓', '↘']
gods_name = ['', 'Apollo', 'Minot', 'Atlas', 'Hepha', 'Artemis', 'Demeter', 'Hermes', 'Pan', 'Athena', 'Prometheus']

def move_to_str(move, player):
	worker, power, move_direction, build_direction = _decode_action(move)
	worker_color = my_workers_color[worker+1] if player == 0 else other_workers_color[worker+1]
	god_power = f' using {gods_name[power]}' if power != NO_GOD else ''

	return f'Move {worker_color}worker {worker+1}{Style.RESET_ALL} to {directions_char[move_direction]} and then build {directions_char[build_direction]}' + god_power


############################# PRINT GAME ######################################

def _print_colors_and_gods(board):
	def god_id(player):
		nonzero = np.flatnonzero(board.gods_power.flat[NB_GODS*player:NB_GODS*(player+1)])
		return gods_name[nonzero[0]] if nonzero.size else 'unk'

	gods_data = board.gods_power[board.gods_power.nonzero()]
	message  = f'Player 0: '
	message += f'{my_workers_color[1]}worker 1  {my_workers_color[2]}worker 2{Style.RESET_ALL} '
	message += f'(has {god_id(0)} power, data={gods_data[0] % 64})    '
	message += f'Player 1: '
	message += f'{other_workers_color[1]}worker 1  {other_workers_color[2]}worker 2{Style.RESET_ALL} '
	message += f'(has {god_id(1)} power, data={gods_data[1] % 64})'
	print(message)

def _print_main(board):
	print(f'-'*11)
	for y in range(5):
		for x in range(5):
			worker, level = board.workers[y, x], board.levels[y, x]
			worker_color = my_workers_color[worker] if worker >= 0 else other_workers_color[-worker]
			if worker != 0 or level > 0:
				print(f'|{worker_color}{levels_char[level]}{Style.RESET_ALL}', end='')
			else:
				print(f'| ', end='')
		print('|')
		print(f'-'*11)

def print_board(board):
	print()
	# _print_round_and_scores(board)
	_print_colors_and_gods(board)
	_print_main(board)