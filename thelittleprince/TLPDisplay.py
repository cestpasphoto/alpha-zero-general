import numpy as np
from colorama import Style, Fore, Back
import random
import itertools

from .TLPGame import NUMBER_PLAYERS
from .TLPLogicNumba import my_unpackbits


def move_to_str(move):
	card_to_take, next_player = divmod(move, NUMBER_PLAYERS)
	card_str = f'take card {card_to_take}'

	if   next_player == 0:
		short_name_next = 'himself'
	elif next_player == 1:
		short_name_next = 'the player on his right'
	elif next_player == NUMBER_PLAYERS-1:
		short_name_next = 'the player on his left'
	elif next_player == 2:
		short_name_next = '2nd player on his right'
	elif next_player == NUMBER_PLAYERS-2:
		short_name_next = '2nd player on his left'
	else: # should not happen
		short_name_next = f'P+{next_player}'
	player_str = f' and choose {short_name_next} as next'
	return card_str + player_str

############################# NAMES ######################################

# Must be synchronized with attributes codes at end of TLPLogicNumba.py
attributes_name = [
	'Card face down', # 0
	'Baobab',         # 1
	'Volcano',        # 2
	'Sunset',         # 3
	'Rose',           # 4
	'Lamp post',      # 5
	'Box',            # 6
	'Big star',       # 7
	'Fox',            # 8
	'Elephant',       # 9
	'Snake',          # 10
	'Sheep (white)',  # 11
	'Sheep (grey)',   # 12
	'Sheep (brown)',  # 13
]

attributes_sign = [
	'🔙', '🌲', '🌋', '🌅', '🌹', '💡', '💼', '🌟', '🦊', '🐘', '🐍', '🐑', '🐺', '🐐'
]

# Must be synchronized with character codes at end of TLPLogicNumba.py
characters_name = [
	'-',                 # 0
	'Vain man',          # 1
	'Geographer',        # 2
	'Astronomer',        # 3
	'King',              # 4
	'Lamplighter',       # 5
	'Hunter',            # 6
	'Drunkard',          # 7
	'W Businessman',     # 8
	'G Businessman',     # 9
	'B Businessman',     # 10
	'Gardener',          # 11
	'Turkish Astronomer',# 12
	'Little Prince',     # 13
]

############################# PRINT GAME ######################################

def _print_round_and_scores(board):
	n = board.num_players
	print('='*10, f' round {board.get_round()}    ', end='')
	for p in range(n):
		print(f'{Style.BRIGHT}P{p}{Style.RESET_ALL}: {board.get_score(p)} points  ', end='')
	print('='*10, Style.RESET_ALL)

def _print_value(value):
	if value > 0:
		print(f' {Style.BRIGHT}{value:>2}{Style.RESET_ALL}', end='')
	else:
		print(f'  0', end='')

def _print_centered_unicode(s, total_size):
	printed_length = len(s.encode('utf-16-le')) // 2
	nb_spaces_pre  = (total_size - printed_length) // 2
	nb_spaces_post = total_size - printed_length - nb_spaces_pre
	for _ in range(nb_spaces_pre):
		print(' ', end='')
	print(s, end='')
	for _ in range(nb_spaces_post):
		print(' ', end='')

def _print_attribute(cell, row, use_two_rows, print_width = 6):
	attributes_list = [attributes_sign[n_attribute] for n_attribute in range(14) if cell[n_attribute] for _ in range(cell[n_attribute])]
	if use_two_rows:
		n = min(len(attributes_list), 6) // 2
		to_print = ''.join(attributes_list[:n]) if row == 0 else ''.join(attributes_list[n:6])
	else:
		to_print = ''.join(attributes_list)
	_print_centered_unicode(to_print, print_width)


def _print_score_details(board):
	for player in range(board.num_players):
		score = board.players_score[player]
		scores_list = [(attributes_sign[n_attribute], score[n_attribute]) for n_attribute in range(1,14) if score[n_attribute]]
		# Consider hack, see TLPLogicNumba:_compute_score()
		if score[0] != 0: # Volcanoes penalty actually wrote in FACE_DOWN
			scores_list.append( (' -', -score[0]) )

		# Now print
		char_count = 0
		for (i, (emoji, sc)) in enumerate(scores_list):
			if i > 0:
				# print(f' + ', end='')
				# char_count += 3
				print(f'  ', end='')
				char_count += 2
			print(f'{emoji}{sc}', end='')
			char_count += 3 if sc < 10 else 4
		print(f' = {score.sum()}', end='')
		char_count += 4 if score.sum() < 10 else 5

		print(' '*(32-char_count), end='')
	print()


def _print_planets(board):
	planet_size = 4
	for planet_y in range(planet_size):
		for row in range(3):
			for player in range(board.num_players):
				for planet_x in range(planet_size):
					cell = board.players_cards[player*16+planet_y*4+planet_x,:]
					if row == 2:      					# border line
						print(f'{Style.DIM} . . . {Style.RESET_ALL}', end='')
					elif cell[14] == 0: 				# Empty cell
						print(f'      {Style.DIM}:{Style.RESET_ALL}', end='')
					elif cell[14] >= 4*25: 				# Corner
						if row == 0:
							character = cell[14] - 4*25
							print(f'{characters_name[character][:6]:>6}', end='')
						else:
							_print_attribute(cell, row, use_two_rows=False)
						print(f'{Style.DIM}:{Style.RESET_ALL}', end='')
					else:
						_print_attribute(cell, row, use_two_rows=True)
						print(f'{Style.DIM}:{Style.RESET_ALL}', end='')
				print('    ', end='')
			print()


def _print_market(board):
	print(f'{Fore.BLACK}{Back.WHITE}MARKET{Style.RESET_ALL} ', end='')
	for i, card in enumerate(board.market):
		if card[14] == 0:
			print(f'{Style.DIM}card {i}:{Style.RESET_ALL}', end='')
			print(' '*12, end='')
		else:
			print(f'card {i}:', end='')
			_print_attribute(card, 0, use_two_rows=False, print_width=12)
		print(f'  ', end='')

	# print('  ', end='')
	print()
	print(f'who can be {Fore.BLACK}{Back.WHITE}next{Style.RESET_ALL}: ', end='')
	who_can_play = my_unpackbits(board.round_and_state[2])
	for p, can_play in enumerate(who_can_play):
		if p == board.round_and_state[1]:
			print(f'{Style.BRIGHT}P{p}{Style.RESET_ALL} ', end='')
		elif can_play:
			print(f'P{p} ', end='')
		else:
			print(f'   ', end='')
	print()


def _print_main(board):
	# Print titles
	print(' '*10, end='')
	for p in range(board.num_players):
		print(f'{Fore.BLACK}{Back.WHITE}Player {p}{Style.RESET_ALL}', end='')
		print(' '*24, end='')
	print()

	_print_planets(board)
	print()
	_print_score_details(board)
	print()
	_print_market(board)

	# for row in board.state.transpose():
	# 	for col in row:
	# 		print(f'{col:2}', end='')
	# 	print()


def print_board(board):
	print()
	_print_round_and_scores(board)
	_print_main(board)