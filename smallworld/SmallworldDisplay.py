import numpy as np
from colorama import Style, Fore, Back
from copy import deepcopy
from SmallworldConstants import *

def move_to_str(move, player):
	return f'Unknown move {move}'

############################# PRINT GAME ######################################

terrains_str = [
	[Back.BLUE          , Fore.WHITE], # WATER
	[Back.LIGHTYELLOW_EX, Fore.BLACK], # SAND
	[Back.YELLOW        , Fore.BLACK], # LAND
	[Back.WHITE         , Fore.BLACK], # MOUNTAIN
	[Back.LIGHTGREEN_EX , Fore.BLACK], # MEADOW
]
powers_str = [' ', '⍎', '⅏']
ppl_str      = [' ', 'A' , 'D' , 'E', 'g', 'G' , 'h', 'H' , 'O' , 'R' , 's', 'S' , 't', 'T' , 'W' , 'p']
ppl_decl_str = [' ', '🄐', '🄓', '🄔', '🄖', '🄖', '🄗', '🄗', '🄞', '🄡', '🄢', '🄢', '🄣', '🄣', '🄦', '🄟']
status_str = [
	'NOT GOOD',
	'starts new turn',
	'just attacked',
	'just abandoned an area',
	'just declined its ppl',
	'will start to redeploy',
	'redeploy ongoing',
	'waiting other player',
]

def generate_background():
	display_matrix = deepcopy(map_display)
	for y in range(DISPLAY_HEIGHT):
		for x in range(DISPLAY_WIDTH):
			area = map_display[y][x]
			terrain = descr[area][0]
			display_matrix[y][x] = deepcopy(terrains_str[terrain])
			display_matrix[y][x].append('.')

	return display_matrix

def add_text(display_matrix, territories):
	for y in range(DISPLAY_HEIGHT):
		for x in range(DISPLAY_WIDTH):
			area = map_display[y][x]
			txt = txt_display[y][x]
			if txt == 1 and territories[area,0] > 0:
				display_matrix[y][x][2] = str(territories[area,0])
				if territories[area,1] >= 0:
					display_matrix[y][x][2] += ppl_str     [ territories[area,1]]
				else:
					display_matrix[y][x][2] += ppl_decl_str[-territories[area,1]]
				if territories[area,3] >= 0:
					display_matrix[y][x][1] += Style.BRIGHT
			elif txt == 2:
				display_matrix[y][x][2] = Fore.LIGHTBLACK_EX + 'a' + str(area)
			elif txt == 3:
				display_matrix[y][x][2] = powers_str[ descr[area][1] ] + ' '
			else:
				display_matrix[y][x][2] = '  '
				
	return display_matrix

def add_legend(display_matrix):
	display_matrix[1].append([Style.RESET_ALL, '', '  '])
	display_matrix[1].append(terrains_str[0] + ['water'])
	display_matrix[1].append([Style.RESET_ALL, '', ' '])
	display_matrix[1].append(terrains_str[1] + ['sand'])
	display_matrix[1].append([Style.RESET_ALL, '', ' '])
	display_matrix[1].append(terrains_str[2] + ['land'])
	display_matrix[1].append([Style.RESET_ALL, '', ' '])
	display_matrix[1].append(terrains_str[3] + ['mountain'])
	display_matrix[1].append([Style.RESET_ALL, '', ' '])
	display_matrix[1].append(terrains_str[4] + ['meadow'])

	legend_power = '  '
	legend_power += powers_str[1] + ' = water source , '
	legend_power += powers_str[2] + ' = mine , '
	display_matrix[2].append([Style.RESET_ALL, '', legend_power])

	legend_ppl = '  '
	# legend_ppl += ppl_str[1] + ' = primitive , '
	# legend_ppl += ppl_str[2] + ' = ogre , '
	display_matrix[3].append([Style.RESET_ALL, '', legend_ppl])

	return display_matrix

def add_players_hand(display_matrix, active_ppl, declined_ppl):
	description = []
	for p in range(NUMBER_PLAYERS):
		description.append([Style.RESET_ALL, '', f'P{p} has {active_ppl[p,0]}ppl "{ppl_str[active_ppl[p,1]]}" (st={status_str[active_ppl[p,2]]})'])
		if declined_ppl[p,1] != NOPPL:
			description[-1][2] += f' and "{ppl_decl_str[-declined_ppl[p,1]]}" on decline'
		description[-1][2] += ' - '
	display_matrix.append(description)
	return display_matrix

def add_scores(display_matrix, scores):
	scores_str = '  Scores: '
	for p in range(NUMBER_PLAYERS):
		scores_str += f' P{p}={scores[p][0]} '
	scores_str += f' turn #{scores[0][1]}'
	display_matrix[4].append([Style.RESET_ALL, '', scores_str])
	return display_matrix

def disp_to_str(display_matrix):
	disp_str = ''
	for y in range(len(display_matrix)):
		for x in range(len(display_matrix[y])):
			bgd, fgd, txt = display_matrix[y][x]
			disp_str += bgd + fgd + txt
		disp_str += Style.RESET_ALL
		disp_str += ('\n' if y < len(display_matrix)-1 else '')
	return disp_str

def print_board(b):
	display_matrix = generate_background()
	display_matrix = add_text(display_matrix, b.territories)
	display_matrix = add_legend(display_matrix)
	display_matrix = add_scores(display_matrix, b.scores)
	display_matrix = add_players_hand(display_matrix, b.active_ppl, b.declined_ppl)
	display_str = disp_to_str(display_matrix)
	print(display_str)

# Used for debug purposes
def print_valids(p, valids_attack, valids_abandon, valids_redeploy, valids_choose, valid_decline):
	print(f'Valids: P{p} can', end='')
	if valids_attack.any():
		print(f' attack area', end='')
		for i in valids_attack.nonzero()[0]:
			print(f' {i}', end='')
		print(', or', end='')

	if valids_abandon.any():
		print(f' abandon area', end='')
		for i in valids_abandon.nonzero()[0]:
			print(f' {i}', end='')
		print(', or', end='')

	if valids_redeploy.any():
		valids_on_each = valids_redeploy[:MAX_REDEPLOY]
		if valids_on_each.any():
			maxi = valids_on_each.nonzero()[0].max()
			if maxi > 0:
				print(f' redeploy up to {maxi}ppl on each area', end='')
			else:
				print(f' skip redeploy', end='')
		else:
			print(f' redeploy on area', end='')
			for i in valids_redeploy.nonzero()[0]:
				print(f' {i-MAX_REDEPLOY}', end='')
		print(', or', end='')

	if valids_choose.any():
		print(f' chose a new people', end='')
		if np.count_nonzero(valids_choose) < 6:
			for i in valids_choose.nonzero()[0]:
				print(f' {i}', end='')
		print(', or', end='')

	if valid_decline:
		print(f' decline current people, or', end='')

	print('.')
