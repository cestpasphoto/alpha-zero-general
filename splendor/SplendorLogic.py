#!/home/best/dev/splendor/venv/bin/python3

import numpy as np
from colorama import Style, Fore, Back
import random
import itertools

def observation_size(num_players):
	return (32+12*num_players, 7)

def action_size():
	return 81

def move_to_str(move, short=False):
	color_names = ['white', 'blue', 'green', 'red', 'black', 'gold']
	if   move < 12:
		tier, index = divmod(move, 4)
		return f'buy tier{tier}-card{index}' if short else f'buy from tier {tier} index {index}'
	elif move < 12+15:
		if move < 12+12:
			tier, index = divmod(move-12, 4)
			return f'rsv t{tier}-c{index}' if short else f'reserve from tier {tier} index {index}'
		else:
			tier = move-12-12
			return f'rsv t{tier}-deck' if short else f'reserve from deck of tier {tier}'
	elif move < 12+15+3:
		index = move-12-15
		return f'buy rsv{index}'if short else f'buy from reserve {index}'
	elif move < 12+15+3+30:
		i = move - 12-15-3
		if i < len(list_different_gems_up_to_3):
			if short:
				gems_str = [ light_colors[i] + "  " + Style.RESET_ALL  for i, v in enumerate(list_different_gems_up_to_3[i][:5]) if v != 0]
				return f'{" ".join(gems_str)}'
			else:
				gems_str = [str(v)+" "+color_names[i] for i, v in enumerate(list_different_gems_up_to_3[i][:5]) if v != 0]
				return f'take {", ".join(gems_str)}'		
		else:
			if short:
				return f'{light_colors[i-len(list_different_gems_up_to_3)] + "    " + Style.RESET_ALL}'
			else:
				return f'take 2 gems of color {color_names[i-len(list_different_gems_up_to_3)]}'
	elif move < 12+15+3+30+20:
		i = move - 12-15-3-30
		if i < len(list_different_gems_up_to_2):
			if short:
				gems_str = [ light_colors[i] + "  " + Style.RESET_ALL for i, v in enumerate(list_different_gems_up_to_2[i][:5]) if v != 0]
				return f'give {" ".join(gems_str)}'
			else:
				gems_str = [str(v)+" "+color_names[i] for i, v in enumerate(list_different_gems_up_to_2[i][:5]) if v != 0]
				return f'give back {", ".join(gems_str)}'
		else:
			if short:
				return f'give {light_colors[i-len(list_different_gems_up_to_2)] + "    " + Style.RESET_ALL}'
			else:
				return f'give back 2 {color_names[i-len(list_different_gems_up_to_2)]}'
	else:
		return f'nothing' if short else f'do nothing'

def row_to_str(row, n=2):
	if row < 1:
		return 'bank'
	if row < 25:
		tier, index = divmod(row-1, 4)
		cost_or_value = ((row-1)%2 == 0)
		return f'Card in tier {tier} index {index//2} ' + ('cost' if cost_or_value else 'value')
	if row < 28:
		return f'Nb cards in deck of tier {row-25}'
	if row < 29+n:
		return f'Nobles num {row-28}'
	if row < 29+2*n:
		return f'Nb of gems of player {row-29-n}/{n}'
	if row < 29+5*n:
		player, index = divmod(row-29-2*n, 3)
		return f'Noble {index} earned by player {player}/{n}'
	if row < 29+6*n:
		return f'Cards of player {row-29-5*n}/{n}'
	if row < 29+12*n:
		player, index = divmod(row-29-6*n, 6)
		cost_or_value = (index%2 == 0)
		return f'Reserve {index//2} of player {player}/{n} ' + ('cost' if cost_or_value else 'value')
	return f'unknown row {row}'

def _gen_list_of_different_gems(max_num_gems):
	gems = [ np.array([int(i==c) for i in range(7)], dtype=np.int8) for c in range(5) ]
	results = []
	for n in range(1, max_num_gems+1):
		results += [ sum(comb) for comb in itertools.combinations(gems, n) ]
	return results


list_different_gems_up_to_3 =  _gen_list_of_different_gems(3)
list_different_gems_up_to_2 =  _gen_list_of_different_gems(2)
np_different_gems_up_to_2 = np.array(list_different_gems_up_to_2, dtype=np.int8)
np_different_gems_up_to_3 = np.array(list_different_gems_up_to_3, dtype=np.int8)

# cards_symmetries = itertools.permutations(range(4))
cards_symmetries   = [(1, 3, 0, 2), (2, 0, 3, 1), (3, 2, 1, 0)]
reserve_symmetries = [
	[], 					# 0 card in reserve
	[], 					# 1 card
	[(1, 0, 2)],			# 2 cards
	[(1, 2, 0), (2, 0, 1)], # 3 cards
]
reserve_symmetries2 = [       # Need constant size to convert to numpy list
	[(-1,-1,-1), (-1,-1,-1)], # 0 card in reserve
	[(-1,-1,-1), (-1,-1,-1)], # 1 card
	[(1, 0, 2) , (-1,-1,-1)], # 2 cards
	[(1, 2, 0) , (2, 0, 1) ], # 3 cards
]
np_cards_symmetries = np.array(cards_symmetries, dtype=np.int8)
np_reserve_symmetries = np.array(reserve_symmetries2, dtype=np.int8)

##### END OF CLASS #####

idx_white, idx_blue, idx_green, idx_red, idx_black, idx_gold, idx_points = range(7)
light_colors = [
	Back.LIGHTWHITE_EX + Fore.BLACK,	# white
	Back.LIGHTBLUE_EX + Fore.WHITE,		# blue
	Back.LIGHTGREEN_EX + Fore.BLACK,	# green
	Back.LIGHTRED_EX + Fore.BLACK,		# red
	Back.LIGHTBLACK_EX + Fore.WHITE,	# black
	Back.LIGHTYELLOW_EX + Fore.BLACK,	# gold
]
strong_colors = [
	Back.WHITE + Fore.BLACK,	# white
	Back.BLUE + Fore.BLACK,		# blue
	Back.GREEN + Fore.BLACK,	# green
	Back.RED + Fore.BLACK,		# red
	Back.BLACK + Fore.WHITE,	# black
	Back.YELLOW + Fore.BLACK,	# gold
]

#    W Blu G  R  Blk  Point
all_nobles = [
	[0, 0, 4, 4, 0, 0, 3],
	[0, 0, 0, 4, 4, 0, 3],
	[0, 4, 4, 0, 0, 0, 3],
	[4, 0, 0, 0, 4, 0, 3],
	[4, 4, 0, 0, 0, 0, 3],
	[3, 0, 0, 3, 3, 0, 3],
	[3, 3, 3, 0, 0, 0, 3],
	[0, 0, 3, 3, 3, 0, 3],
	[0, 3, 3, 3, 0, 0, 3],
	[3, 3, 0, 0, 3, 0, 3],
]
np_all_nobles  = np.array(all_nobles , dtype=np.int8)

#             COST                      GAIN
#         W Blu G  R  Blk        W Blu G  R  Blk  Point
all_cards_1 = [
	[
		[[0, 0, 0, 0, 3, 0, 0], [0, 1, 0, 0, 0, 0, 0]],
		[[1, 0, 0, 0, 2, 0, 0], [0, 1, 0, 0, 0, 0, 0]],
		[[0, 0, 2, 0, 2, 0, 0], [0, 1, 0, 0, 0, 0, 0]],
		[[1, 0, 2, 2, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]],
		[[0, 1, 3, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]],
		[[1, 0, 1, 1, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0]],
		[[1, 0, 1, 2, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0]],
		[[0, 0, 0, 4, 0, 0, 0], [0, 1, 0, 0, 0, 0, 1]],
	],
	[
		[[3, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
		[[0, 2, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
		[[2, 0, 0, 2, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
		[[2, 0, 1, 0, 2, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
		[[1, 0, 0, 1, 3, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
		[[1, 1, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
		[[2, 1, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
		[[4, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 1]],
	],
	[
		[[0, 0, 3, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]],
		[[0, 0, 2, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]],
		[[2, 0, 2, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]],
		[[2, 2, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]],
		[[0, 0, 1, 3, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0]],
		[[1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]],
		[[1, 2, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]],
		[[0, 4, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 1]],
	],
	[
		[[0, 3, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
		[[0, 0, 0, 2, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
		[[0, 2, 0, 0, 2, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
		[[0, 2, 2, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
		[[3, 1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
		[[0, 1, 1, 1, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
		[[0, 1, 2, 1, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
		[[0, 0, 4, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 1]],
	],
	[
		[[0, 0, 0, 3, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]],
		[[2, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]],
		[[0, 2, 0, 2, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]],
		[[0, 1, 0, 2, 2, 0, 0], [0, 0, 1, 0, 0, 0, 0]],
		[[1, 3, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]],
		[[1, 1, 0, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0]],
		[[1, 1, 0, 1, 2, 0, 0], [0, 0, 1, 0, 0, 0, 0]],
		[[0, 0, 0, 0, 4, 0, 0], [0, 0, 1, 0, 0, 0, 1]],
	],
]

#             COST                      GAIN
#         W Blu G  R  Blk        W Blu G  R  Blk  Point
all_cards_2 = [
	[
		[[0, 2, 2, 3, 0, 0, 0], [0, 1, 0, 0, 0, 0, 1]],
		[[0, 2, 3, 0, 3, 0, 0], [0, 1, 0, 0, 0, 0, 1]],
		[[0, 5, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 2]],
		[[5, 3, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 2]],
		[[2, 0, 0, 1, 4, 0, 0], [0, 1, 0, 0, 0, 0, 2]],
		[[0, 6, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 3]],
	],
	[
		[[2, 0, 0, 2, 3, 0, 0], [0, 0, 0, 1, 0, 0, 1]],
		[[0, 3, 0, 2, 3, 0, 0], [0, 0, 0, 1, 0, 0, 1]],
		[[0, 0, 0, 0, 5, 0, 0], [0, 0, 0, 1, 0, 0, 2]],
		[[3, 0, 0, 0, 5, 0, 0], [0, 0, 0, 1, 0, 0, 2]],
		[[1, 4, 2, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 2]],
		[[0, 0, 0, 6, 0, 0, 0], [0, 0, 0, 1, 0, 0, 3]],
	],
	[
		[[3, 2, 2, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 1]],
		[[3, 0, 3, 0, 2, 0, 0], [0, 0, 0, 0, 1, 0, 1]],
		[[5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 2]],
		[[0, 0, 5, 3, 0, 0, 0], [0, 0, 0, 0, 1, 0, 2]],
		[[0, 1, 4, 2, 0, 0, 0], [0, 0, 0, 0, 1, 0, 2]],
		[[0, 0, 0, 0, 6, 0, 0], [0, 0, 0, 0, 1, 0, 3]],
	],
	[
		[[0, 0, 3, 2, 2, 0, 0], [1, 0, 0, 0, 0, 0, 1]],
		[[2, 3, 0, 3, 0, 0, 0], [1, 0, 0, 0, 0, 0, 1]],
		[[0, 0, 0, 5, 0, 0, 0], [1, 0, 0, 0, 0, 0, 2]],
		[[0, 0, 0, 5, 3, 0, 0], [1, 0, 0, 0, 0, 0, 2]],
		[[0, 0, 1, 4, 2, 0, 0], [1, 0, 0, 0, 0, 0, 2]],
		[[6, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 3]],
	],
	[
		[[2, 3, 0, 0, 2, 0, 0], [0, 0, 1, 0, 0, 0, 1]],
		[[3, 0, 2, 3, 0, 0, 0], [0, 0, 1, 0, 0, 0, 1]],
		[[0, 0, 5, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 2]],
		[[0, 5, 3, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 2]],
		[[4, 2, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 2]],
		[[0, 0, 6, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 3]],
	]
]

#             COST                      GAIN
#         W Blu G  R  Blk        W Blu G  R  Blk  Point
all_cards_3 = [
	[
		[[3, 0, 3, 3, 5, 0, 0], [0, 1, 0, 0, 0, 0, 3]],
		[[7, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 4]],
		[[6, 3, 0, 0, 3, 0, 0], [0, 1, 0, 0, 0, 0, 4]],
		[[7, 3, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 5]],
	],
	[
		[[3, 5, 3, 0, 3, 0, 0], [0, 0, 0, 1, 0, 0, 3]],
		[[0, 0, 7, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 4]],
		[[0, 3, 6, 3, 0, 0, 0], [0, 0, 0, 1, 0, 0, 4]],
		[[0, 0, 7, 3, 0, 0, 0], [0, 0, 0, 1, 0, 0, 5]],
	],
	[
		[[3, 3, 5, 3, 0, 0, 0], [0, 0, 0, 0, 1, 0, 3]],
		[[0, 0, 0, 7, 0, 0, 0], [0, 0, 0, 0, 1, 0, 4]],
		[[0, 0, 3, 6, 3, 0, 0], [0, 0, 0, 0, 1, 0, 4]],
		[[0, 0, 0, 7, 3, 0, 0], [0, 0, 0, 0, 1, 0, 5]],
	],
	[
		[[0, 3, 3, 5, 3, 0, 0], [1, 0, 0, 0, 0, 0, 3]],
		[[0, 0, 0, 0, 7, 0, 0], [1, 0, 0, 0, 0, 0, 4]],
		[[3, 0, 0, 3, 6, 0, 0], [1, 0, 0, 0, 0, 0, 4]],
		[[3, 0, 0, 0, 7, 0, 0], [1, 0, 0, 0, 0, 0, 5]],
	],
	[
		[[5, 3, 0, 3, 3, 0, 0], [0, 0, 1, 0, 0, 0, 3]],
		[[0, 7, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 4]],
		[[3, 6, 3, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 4]],
		[[0, 7, 3, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 5]],
	]
]

all_cards = [all_cards_1, all_cards_2, all_cards_3]
np_all_cards_1 = np.array(all_cards_1, dtype=np.int8)
np_all_cards_2 = np.array(all_cards_2, dtype=np.int8)
np_all_cards_3 = np.array(all_cards_3, dtype=np.int8)
len_all_cards = np.array([len(all_cards_1[0]), len(all_cards_2[0]), len(all_cards_3[0])], dtype=np.int8)

def _print_round_and_scores(board):
	n = board.num_players
	print('='*10*n, f' round {board.get_round()}    ', end='')
	for p in range(n):
		print(f'{Style.BRIGHT}P{p}{Style.RESET_ALL}: {board.get_score(p)} points  ', end='')
	print('='*10*n, Style.RESET_ALL)

def _print_nobles(board):
	print(f'{Style.BRIGHT}Nobles:  {Style.RESET_ALL}', end='')
	for noble in board.nobles:
		if noble[idx_points] == 0:
			print(f'< {Style.DIM}empty{Style.RESET_ALL} >', end=' ')
		else:
			print(f'< {noble[idx_points]} points ', end='')
			for i, color in enumerate(light_colors):
				if noble[i] != 0:
					print(f'{color} {noble[i]} {Style.RESET_ALL} ', end='')
			print(f'> ', end='')
	print(f'{Style.RESET_ALL}')

def _print_card_line(card, line, space_between):
	if card[1,:5].sum() == 0:
		print(f' '*(8+space_between), end='')
		return
	card_color = np.flatnonzero(card[1,:5] != 0)[0]
	background = light_colors[card_color]
	print(background, end= '')
	if line == 0:
		print(f'     {Style.BRIGHT}{card[1][idx_points]}{Style.NORMAL}  ', end='')
	else:
		card_cost = np.flatnonzero(card[0,:5] != 0)
		if line-1 < card_cost.size:
			color = card_cost[line-1]
			value = card[0,color]
			print(f' {light_colors[color]} {value} {background}    ', end='')
		else:
			print(' '*8, end='')
	print(Style.RESET_ALL, end=' '*space_between)

def _print_tiers(board):
	for tier in range(2, -1, -1):
		for line in range(5):
			if line == 3:
				print(f'Tier {tier}:  ', end='')
			elif line == 4 :
				print(f'  ({board.nb_deck_tiers[2*tier].sum():>2})   ', end='')
			else:
				print(f'         ', end='')
			for i in range(4):
				_print_card_line(board.cards_tiers[8*tier+2*i:8*tier+2*i+2, :], line, 4)
			print()
		print()

def _print_bank(board):
	print(f'{Style.BRIGHT}Bank: {Style.RESET_ALL}   ', end='')
	for c in range(6):
		print(f'{light_colors[c]} {board.bank[0][c]} {Style.RESET_ALL} ', end='')
	print(f'{Style.RESET_ALL}')

def _print_players(board):
	n = board.num_players
	# NAMES
	print(' '*19, end='')
	for p in range(n):
		print(f'Player {p}', end='')
		if p < n-1:
			print(f' '*26, end='')
	print()

	# NOBLES
	print(' '*9, end='')
	for p in range(n):
		for noble in board.players_nobles[3*p:3*p+3]:
			if noble[idx_points] > 0:
				print(f'  < {Style.BRIGHT}{noble[idx_points]}{Style.RESET_ALL} >  ', end='')
			else:
				print(f'        ', end='')
		print(f' '*10, end='')
	print()

	# GEMS
	print(f'{Style.BRIGHT}Gems: {Style.RESET_ALL}   ', end='')
	for p in range(n):
		for c in range(6):
			my_gems  = board.players_gems[p][c]
			print(f'{light_colors[c]} {my_gems} {Style.RESET_ALL} ', end='')
		print(f' Σ{board.players_gems[p].sum():2}      ', end='')
	print()

	# CARDS
	# print()
	print(f'{Style.BRIGHT}Cards: {Style.RESET_ALL}  ', end='')
	for p in range(n):
		for c in range(5):
			my_cards = board.players_cards[p][c]
			print(f'{light_colors[c]} {my_cards} {Style.RESET_ALL} ', end='')
		print(f'              ', end='')
	print()

	# RESERVED
	if board.players_reserved.sum() > 0:
		print()
		for line in range(5):
			if line == 2:
				print(f'{Style.BRIGHT}Reserve: {Style.RESET_ALL}', end='')
			else:
				print(' '*9, end='')
			for p in range(n):
				for r in range(3):
					reserved = board.players_reserved[6*p+2*r:6*p+2*r+2]
					if reserved[0].sum() != 0:
						_print_card_line(reserved, line, 2)
					else:
						print(f' '*10, end='')
				print(f' '*4, end='')
			print()

def print_board(board):
	_print_round_and_scores(board)
	_print_nobles(board)
	print()
	_print_tiers(board)
	_print_bank(board)
	print()
	_print_players(board)
