#!/home/best/dev/splendor/venv/bin/python3

import numpy as np
from colorama import Style, Fore, Back
import random
import itertools

def move_to_str(move):
	if   move < 15:
		return f'buy {cards_name[move]}'
	elif move < 15+4:
		i = move-15
		return f'enable {monuments_name[i]}'
	elif move == 15+4:
		return f'roll dice(s) again'
	else:
		return f'do nothing'

############################# NAMES ######################################

cards_name = [
	'champs de blé',              # 0
	'ferme',                      # 1
	'boulangerie',                # 2
	'café',                       # 3
	'supérette',                  # 4
	'forêt',                      # 5
	'stade',                      # 6
	'centre d\'affaires',         # 7
	'chaîne de télévision',       # 8
	'fromagerie',                 # 9
	'fabrique de meubles',        # 10
	'mine',                       # 11
	'restaurant',                 # 12
	'verger',                     # 13
	'marché de fruits & légumes', # 14
]

cards_short_descr = [
	('  +1  ', ' 1 ALL'), # 0
	('  +1  ', ' 2 ALL'), # 1
	('  +1  ', '2-3 ME'), # 2
	('  -1  ', '3 GIVE'), # 3
	('  +3  ', ' 4  ME'), # 4
	('  +1  ', ' 5 ALL'), # 5
	('ALL -2', ' 6  ME'), # 6
	('swap c', ' 6  ME'), # 7
	('+5 pla', ' 6  ME'), # 8
	('+3*cow', ' 7  ME'), # 9
	('+3*gea', ' 8  ME'), # 10
	('  +5  ', ' 9 ALL'), # 11
	('  -2  ', '9-10 G'), # 12
	('  +3  ', '10 ALL'), # 13
	('+2*whe', '11-2ME'), # 14
]

monuments_name = [
	'gare',                # 0
	'centre commercial',   # 1
	'tour radio',          # 2
	'parc d\'attractions', # 3
]

############################# PRINT GAME ######################################

def _print_round_and_scores(board):
	n = board.num_players
	print('='*20, f' round {board.get_round()}    ', end='')
	for p in range(n):
		print(f'{Style.BRIGHT}P{p}{Style.RESET_ALL}: {board.get_score(p)} points  ', end='')
	print('='*20, Style.RESET_ALL)

def _print_titles(board):
	for line in range(2):
		print(' '*7, end='')
		for descr in cards_short_descr:
			print(descr[line]+' ', end='')
		if line == 1:
			for mon in range(4):
				print(f'M{mon} ', end='')
		print()

def _print_market(board):
	print(f'{Style.BRIGHT}Market:{Style.RESET_ALL}', end='')
	for c in board.market[:,0]:
		if c == 0:
			print(f'   {Style.DIM}0{Style.RESET_ALL}   ', end='')
		else:
			print(f'   {c}   ', end='')
	print(f'{Style.RESET_ALL}')

def _print_players(board, p):
	print(f'{Style.BRIGHT}P{p}:{Style.RESET_ALL} {board.players_money[p,0]:2}$', end='')
	for c in board.players_cards[15*p:15*(p+1),0]:
		if c == 0:
			print(f'   {Style.DIM}0{Style.RESET_ALL}   ', end='')
		else:
			print(f'   {c}   ', end='')

	for c in board.players_monuments[4*p:4*(p+1),0]:
		if c == 0:
			print(f' {Style.DIM}0{Style.RESET_ALL} ', end='')
		else:
			print(f' {c} ', end='')
	print(f'{Style.RESET_ALL}')


def print_board(board):
	print()
	_print_round_and_scores(board)
	_print_titles(board)
	print()
	_print_market(board)
	print()
	for p in range(board.num_players):
		_print_players(board, p)
	print()
