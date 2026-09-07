import numpy as np
from colorama import Style, Fore, Back
from copy import deepcopy
from .SmallworldConstants import *
from .SmallworldMaps import *

############################# PRINT GAME ######################################

terrains_str = [
	[Back.GREEN         , Fore.BLACK], # FORESTT
	[Back.LIGHTYELLOW_EX, Fore.BLACK], # FARMLAND
	[Back.LIGHTGREEN_EX , Fore.BLACK], # HILLT
	[Back.LIGHTRED_EX   , Fore.BLACK], # SWAMPT
	[Back.WHITE         , Fore.BLACK], # MOUNTAIN
	[Back.LIGHTBLUE_EX  , Fore.WHITE], # WATER
]
powers_str = [' ', '⅏','ℵ', '⍎']

ppl_str      = [' ', 'A' , 'D' , 'E', 'g', 'G' , 'h', 'H' , 'O' , 'R' , 's', 'S' , 't', 'T' , 'W' , 'l']
ppl_decl_str = [' ', '🄐', '🄓', '🄔', '🄖', '🄖', '🄗', '🄗', '🄞', '🄡', '🄢', '🄢', '🄣', '🄣', '🄦', '🄛']
ppl_long_str = [' ', 'AMAZON','DWARF','ELF','GHOUL','GIANT','HALFLING','HUMAN','ORC','RATMAN','SKELETON','SORCERER','TRITON','TROLL','WIZARD', 'LOST_TRIBE']
power_long_str = [' ','ALCHEMIST','BERSERK','BIVOUACKING','COMMANDO','DIPLOMAT','DRAGONMASTER', 'FLYING','FOREST','FORTIFIED','HEROIC','HILL','MERCHANT','MOUNTED','PILLAGING','SEAFARING','SPIRIT','STOUT','SWAMP','UNDERWORLD','WEALTHY']
ac_or_dec_str = ['decline-spirit ppl', 'decline ppl', 'active ppl', '']
status_str = [
	'',
	'is ready to play',
	'chose new ppl',
	'abandoned',
	'attacked',
	'attacked with dice',
	'forced to abandon (amazon)',
	'redeployed',
	'to decline (stout)',
	'is waiting',
]

last_board, last_board_already_displayed = None, False

def generate_background():
	display_matrix = deepcopy(map_display)
	for y in range(DISPLAY_HEIGHT):
		for x in range(DISPLAY_WIDTH):
			area, _ = map_display[y][x]
			terrain = descr[area][0]
			display_matrix[y][x] = deepcopy(terrains_str[terrain])
			display_matrix[y][x].append('.')

	return display_matrix

def add_text(display_matrix, territories):
	for y in range(DISPLAY_HEIGHT):
		for x in range(DISPLAY_WIDTH):
			area, txt = map_display[y][x]
			if txt == 1 and territories[area,0] > 0:
				display_matrix[y][x][2] = str(territories[area,0])
				if territories[area,1] >= 0:
					display_matrix[y][x][2] += ppl_str     [ territories[area,1]]
					display_matrix[y][x][1] += Style.BRIGHT
				else:
					display_matrix[y][x][2] += ppl_decl_str[-territories[area,1]]					
			elif txt == 2:
				display_matrix[y][x][2] = Fore.LIGHTBLACK_EX + f'{area:2}'
			elif txt == 3:
				display_matrix[y][x][2] = ''
				for i in range(1, 4):
					if descr[area][i]:
						display_matrix[y][x][2] += powers_str[i]
				display_matrix[y][x][2] += ' ' * (2-len(display_matrix[y][x][2]))
			elif txt == 4 and territories[area, 3:5].sum() > 0:
				if territories[area, 3:5].sum() >= IMMUNITY:
					display_matrix[y][x][2] = '**'
				else:
					display_matrix[y][x][2] = '+' + str(territories[area, 3:5].sum())
			else:
				display_matrix[y][x][2] = '  '
				
	return display_matrix

def add_legend(display_matrix, peoples):
	display_matrix[0].append([Style.RESET_ALL, '', '  '])
	display_matrix[0].append(terrains_str[0] + ['forest'])
	display_matrix[0].append([Style.RESET_ALL, '', ' '])
	display_matrix[0].append(terrains_str[1] + ['farmland'])
	display_matrix[0].append([Style.RESET_ALL, '', ' '])
	display_matrix[0].append(terrains_str[2] + ['hill'])
	display_matrix[0].append([Style.RESET_ALL, '', ' '])
	display_matrix[0].append(terrains_str[3] + ['swamp'])
	display_matrix[0].append([Style.RESET_ALL, '', ' '])
	display_matrix[0].append(terrains_str[4] + ['mountain'])

	legend_power = '  '
	legend_power += powers_str[1] + ' = cavern , '
	legend_power += powers_str[2] + ' = magic , '
	legend_power += powers_str[3] + ' = mine , '
	display_matrix[1].append([Style.RESET_ALL, '', legend_power])

	legend_ppl = '  '
	for i in range(NUMBER_PLAYERS):
		for j in range(3):
			ppl, power, pplinfo, powerinfo = abs(peoples[i,j,1:5])
			if ppl != NOPPL:
				short_str = ppl_str[ppl] if j == ACTIVE else ppl_decl_str[ppl]
				legend_ppl += f'{short_str} = {ppl_long_str[ppl]}'
				if power != NOPOWER:
					legend_ppl += f'+{power_long_str[power]}'
				if pplinfo != 0 or powerinfo != 0:
					pplinfo_str = str(pplinfo) if pplinfo < 64 else (str(pplinfo%64)+'*')
					pwrinfo_str = str(powerinfo) if powerinfo < 64 else (str(powerinfo%64)+'*')
					legend_ppl += f' ({pplinfo_str}-{pwrinfo_str})'
				legend_ppl += f', '
	display_matrix[2].append([Style.RESET_ALL, '', legend_ppl])

	return display_matrix

def add_players_status(display_matrix, peoples, round_status, game_status):
	for p in range(NUMBER_PLAYERS):
		description = f'  P{p}: sc={game_status[p,6]+SCORE_OFFSET:2} #{game_status[p,3]} netwdt={round_status[p,3]}'
		description += f' - has {peoples[p,ACTIVE,0]}ppl "{ppl_str[peoples[p,ACTIVE,1]]}"'
		if peoples[p,DECLINED,1] != NOPPL:
			description += f' and "{ppl_decl_str[-peoples[p,DECLINED,1]]}"'
		if peoples[p,DECLINED_SPIRIT,1] != NOPPL:
			description += f' and "{ppl_decl_str[-peoples[p,DECLINED_SPIRIT,1]]}"'
		if round_status[p, 4] != PHASE_WAIT:
			description += f', {ac_or_dec_str[game_status[p, 4]]} {status_str[round_status[p, 4]]}'
		display_matrix[6+p].append([Style.RESET_ALL, '', description])
	return display_matrix

def add_deck(display_matrix, visible_deck):
	for index, range_beg, range_end in [(3, 0, DECK_SIZE//2), (4, DECK_SIZE//2, DECK_SIZE)]:
		deck_str = f'  Deck:' if index == 3 else f'       '
		for i in range(range_beg, range_end):
			nb, ppl, power, coins = visible_deck[i,0], visible_deck[i,1], visible_deck[i,2], visible_deck[i,6]
			description = f'{nb}x{ppl_long_str[ppl].lower()[:8]}-{power_long_str[power].lower()[:8]}'
			if coins > 0:
				description += f'+{coins}'
			deck_str += f' {i} = {description:22}'
		display_matrix[index].append([Style.RESET_ALL, Style.DIM, deck_str])

	return display_matrix

def disp_to_str(display_matrix):
	disp_str = ''
	for y in range(len(display_matrix)):
		for x in range(len(display_matrix[y])):
			bgd, fgd, txt = display_matrix[y][x]
			disp_str += bgd + fgd + txt + Style.RESET_ALL
		disp_str += ('\n' if y < len(display_matrix)-1 else '')
	return disp_str

def which_board_to_print(prev_board, cur_board):
	if prev_board is None:
		return cur_board

	prev_player = np.argwhere(prev_board.round_status[:, 4] != PHASE_WAIT)[0][0]
	prev_phase, cur_phase = prev_board.round_status[prev_player, 4], cur_board.round_status[prev_player, 4]
	if prev_phase == cur_phase or (prev_phase, cur_phase) in [(PHASE_ABANDON, PHASE_CONQUEST)]:
		return None
	if cur_phase in [PHASE_CHOOSE, PHASE_CONQ_WITH_DICE, PHASE_WAIT]:
		return cur_board
	return prev_board

def print_board(b):
	global last_board, last_board_already_displayed
	board_to_print = which_board_to_print(last_board, b)
	if board_to_print is not None and not (last_board is not None and np.array_equal(board_to_print.state, last_board.state) and last_board_already_displayed):
		display_matrix = generate_background()
		display_matrix = add_text(display_matrix, board_to_print.territories)
		display_matrix = add_legend(display_matrix, board_to_print.peoples)
		display_matrix = add_deck(display_matrix, board_to_print.visible_deck)
		display_matrix = add_players_status(display_matrix, board_to_print.peoples, board_to_print.round_status, board_to_print.game_status)
		
		display_str = disp_to_str(display_matrix)
		print(display_str)

		last_board_already_displayed = np.array_equal(board_to_print.state, b.state)		
	else:
		last_board_already_displayed = False
	last_board = deepcopy(b)

# Used for debug purposes
def print_valids(p, valids_attack, valids_special, valids_abandon, valids_redeploy, valids_specialpwr, valids_choose, valid_decline, valid_end):
	print(f'Valids: P{p} can', end='')
	if valids_attack.any():
		print(f' attack area', end='')
		for i in valids_attack.nonzero()[0]:
			print(f' {i}', end='')
		print(', or', end='')

	if valids_special.any():
		print(f' specialPPL on', end='')
		for i in valids_special.nonzero()[0]:
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

	if valids_specialpwr.any():
		print(f' specialPWR on', end='')
		for i in valids_specialpwr.nonzero()[0]:
			print(f' {i}', end='')
		print(', or', end='')

	if valids_choose.any():
		print(f' chose a new people', end='')
		if np.count_nonzero(valids_choose) < 6:
			for i in valids_choose.nonzero()[0]:
				print(f' {i}', end='')
		print(', or', end='')

	if valid_decline:
		print(f' decline current people, or', end='')

	if valid_end:
		print(f' end turn', end='')

	print('.')

def move_to_str(move, player=0):
	if   move < NB_AREAS:
		area = move
		return f'Abandon area {area}'
	elif move < 2*NB_AREAS:
		area = move - NB_AREAS
		return f'Attack area {area}'
	elif move < 3*NB_AREAS:
		area = move - 2*NB_AREAS
		return f'People capacity on area {area}'
	elif move < 4*NB_AREAS:
		area = move - 3*NB_AREAS
		return f'Power capacity on area {area}'
	elif move < 5*NB_AREAS+MAX_REDEPLOY:
		param = move - 4*NB_AREAS
		if param == 0:
			return f'Skip redeploy'
		elif param < MAX_REDEPLOY:
			return f'Redeploy {param}ppl on EACH of your areas'
		else:
			return f'Redeploy 1ppl on area {param-MAX_REDEPLOY}'
	elif move < 5*NB_AREAS+MAX_REDEPLOY+DECK_SIZE:
		slot = move - 5*NB_AREAS-MAX_REDEPLOY
		return f'Choose deck slot {slot}'
	elif move < 5*NB_AREAS+MAX_REDEPLOY+DECK_SIZE+1:
		return f'Decline'
	elif move < 5*NB_AREAS+MAX_REDEPLOY+DECK_SIZE+2:
		return f'End turn'
	else:
		return f'Unknown move {move}'


############################# MOVE LIST FOR A HUMAN ###########################
#
# describe_moves(state, valids) turns the raw action indices into a grouped,
# annotated list. Display only: it never mutates the state and is never called
# by the engine, so a wrong hint costs a confused human, never a wrong game.
#
# The attack-cost hint MIRRORS Board._minimum_ppl_for_attack(); if that method
# changes, this must be updated too (it is marked with ~ so it reads as an
# estimate). Everything else is read straight from the state array.

terrain_long_str = ['forest', 'farmland', 'hill', 'swamp', 'mountain', 'water']

# Powers that expose a "power capacity" action, and what it does on an area.
_PWR_ACTION_STR = {
	BIVOUACKING : 'place a campment (+1 def)',
	FORTIFIED   : 'place a fortress (+1 def, +1 pt)',
	HEROIC      : 'place a hero (full immunity)',
	DIPLOMAT    : 'make peace with the people there',
	DRAGONMASTER: 'dragon attack (auto-win, full immunity)',
}


def split_state(state):
	"""Views on a raw (nb_vect, 8) state array. Mirrors Board.copy_state()."""
	A, P = NB_AREAS, NUMBER_PLAYERS
	territories  = state[0                : A]
	peoples      = state[A                : A+3*P].reshape((P, 3, 8))
	visible_deck = state[A+3*P            : A+3*P+DECK_SIZE]
	round_status = state[A+3*P+DECK_SIZE  : A+4*P+DECK_SIZE]
	game_status  = state[A+4*P+DECK_SIZE  : A+5*P+DECK_SIZE]
	return territories, peoples, visible_deck, round_status, game_status


def _area_str(territories, area):
	"""Who holds this area and how well it is defended."""
	nb, ppl, pwr, owner = territories[area, 0], territories[area, 1], territories[area, 2], territories[area, 7]
	terrain = terrain_long_str[descr[area][0]]
	extra = ''.join(s for flag, s in [(CAVERN, ' cavern'), (MAGIC, ' magic'), (MINE, ' mine')] if descr[area][flag])
	if ppl == NOPPL:
		who = 'empty'
	elif ppl == LOST_TRIBE:
		who = f'lost tribe x{nb}'
	else:
		name = ppl_long_str[abs(ppl)].lower()
		decl = '' if ppl > 0 else ' (declined)'
		pwr_s = f'+{power_long_str[abs(pwr)].lower()}' if pwr != NOPOWER else ''
		who = f'P{owner} {name}{pwr_s} x{nb}{decl}'
	return f'{terrain:8}{extra:<14} {who:<38} def {territories[area, 5]}'


def _attack_cost(territories, area, current_ppl):
	"""Mirror of Board._minimum_ppl_for_attack(). Display hint only."""
	cost = int(territories[area, 5]) + 2
	ppl, pwr = current_ppl[1], current_ppl[2]
	neighbours = connexity_matrix[area].astype(bool)
	borders = lambda terrain: bool((descr[neighbours, 0] == terrain).any())
	if ppl == TRITON and borders(WATER):
		cost -= 1
	if ppl == GIANT and borders(MOUNTAIN):
		cost -= 1
	if pwr == COMMANDO:
		cost -= 1
	if pwr == MOUNTED and descr[area][0] in [HILLT, FARMLAND]:
		cost -= 1
	if pwr == UNDERWORLD and descr[area][CAVERN]:
		cost -= 1
	return max(cost, 1)


def state_to_str(state):
	"""Render a raw (nb_vect, 8) state array.

	print_board() takes a Board object, keeps a module-level cache of the
	previous position to skip redundant frames, and deepcopy()s that Board --
	which fails on a numba jitclass. This renders straight from the array:
	no cache, no Board, no deepcopy, so it is safe to call on any stored state
	in any order (log replay, diagnostics).
	"""
	territories, peoples, visible_deck, round_status, game_status = split_state(state)
	display_matrix = generate_background()
	display_matrix = add_text(display_matrix, territories)
	display_matrix = add_legend(display_matrix, peoples)
	display_matrix = add_deck(display_matrix, visible_deck)
	display_matrix = add_players_status(display_matrix, peoples, round_status, game_status)
	return disp_to_str(display_matrix)


def move_detail(state, move):
	"""One-line rich description of a single move, for ranking tables.
	Shares _area_str / _attack_cost with describe_moves(), so the parts that
	encode a rule live in exactly one place."""
	territories, peoples, visible_deck, round_status, game_status = split_state(state)
	A, MR = NB_AREAS, MAX_REDEPLOY
	current_id = int(game_status[0, 4])
	if current_id < 0:
		current_id = ACTIVE
	current_ppl = peoples[0, current_id, :]

	if move < A:
		return f'Abandon  area {move:2}  {_area_str(territories, move)}'
	if move < 2*A:
		area = move - A
		return (f'Attack   area {area:2}  {_area_str(territories, area)}'
		        f'   cost ~{_attack_cost(territories, area, current_ppl)}')
	if move < 3*A:
		area = move - 2*A
		return f'PplCap   area {area:2}  {_area_str(territories, area)}'
	if move < 4*A:
		area = move - 3*A
		return f'PwrCap   area {area:2}  {_area_str(territories, area)}'
	if move < 5*A + MR:
		param = move - 4*A
		if param == 0:
			return 'Redeploy skip (leave your people where they are)'
		if param < MR:
			return f'Redeploy {param} more ppl on EACH area you own'
		area = param - MR
		return f'Redeploy 1 ppl on area {area:2}  {_area_str(territories, area)}'
	if move < 5*A + MR + DECK_SIZE:
		slot = move - (5*A + MR)
		nb, ppl, pwr, coins = (int(visible_deck[slot, i]) for i in (0, 1, 2, 6))
		combo = f'{nb:2}x{ppl_long_str[ppl].lower()}+{power_long_str[pwr].lower()}'
		return f'Choose   slot {slot}  {combo:<34}pay {slot}, get {coins}  -> score {coins - slot:+d}'
	if move == 5*A + MR + DECK_SIZE:
		return 'DECLINE your active people'
	if move == 5*A + MR + DECK_SIZE + 1:
		return 'END your turn'
	return f'Unknown move {move}'


def describe_moves(state, valids):
	"""Returns a list of printable lines describing every legal move."""
	territories, peoples, visible_deck, round_status, game_status = split_state(state)
	A, MR = NB_AREAS, MAX_REDEPLOY
	current_id = int(game_status[0, 4])
	if current_id < 0:
		current_id = ACTIVE
	current_ppl = peoples[0, current_id, :]
	in_hand = int(current_ppl[0])
	score = int(game_status[0, 6]) + SCORE_OFFSET
	legal = [int(a) for a in np.flatnonzero(valids)]

	ppl_name = ppl_long_str[abs(current_ppl[1])].lower() if current_ppl[1] != NOPPL else 'no people'
	pwr_name = power_long_str[abs(current_ppl[2])].lower() if current_ppl[2] != NOPOWER else ''
	head = f'you are P0: {ppl_name}{"+" + pwr_name if pwr_name else ""}'
	head += f' [{ac_or_dec_str[current_id]}], {in_hand} ppl in hand, score {score}'
	lines = [head, '']

	def block(title, items):
		if items:
			lines.append(title)
			lines.extend(items)
			lines.append('')

	choose = []
	for a in legal:
		if not (5*A+MR <= a < 5*A+MR+DECK_SIZE):
			continue
		slot = a - (5*A + MR)
		nb, ppl, pwr, coins = (int(visible_deck[slot, i]) for i in (0, 1, 2, 6))
		combo = f'{nb:2}x{ppl_long_str[ppl].lower()}+{power_long_str[pwr].lower()}'
		choose.append(f'  {a:4}  slot {slot}  {combo:<34}'
		              f'pay {slot}, get {coins}  -> score {coins - slot:+d}')
	block('CHOOSE a new people (you pay the slot number, you collect the coins on it):', choose)

	block(f'ATTACK  (you have {in_hand} ppl in hand; cost ~= defense + 2, minus your bonuses):', [
		f'  {a:4}  area {a-A:2}  {_area_str(territories, a-A)}'
		f'   cost ~{_attack_cost(territories, a-A, current_ppl)}'
		for a in legal if A <= a < 2*A])

	block('ABANDON one of your areas:', [
		f'  {a:4}  area {a:2}  {_area_str(territories, a)}'
		for a in legal if a < A])

	if any(2*A <= a < 3*A for a in legal):
		what = 'replace the lone enemy token there by one of yours' if current_ppl[1] == SORCERER else 'people capacity'
		block(f'PEOPLE CAPACITY ({ppl_name}): {what}', [
			f'  {a:4}  area {a-2*A:2}  {_area_str(territories, a-2*A)}'
			for a in legal if 2*A <= a < 3*A])

	if any(3*A <= a < 4*A for a in legal):
		what = _PWR_ACTION_STR.get(abs(int(current_ppl[2])), 'power capacity')
		block(f'POWER CAPACITY ({pwr_name}): {what}', [
			f'  {a:4}  area {a-3*A:2}  {_area_str(territories, a-3*A)}'
			for a in legal if 3*A <= a < 4*A])

	redeploy = []
	for a in legal:
		if not (4*A <= a < 5*A+MR):
			continue
		param = a - 4*A
		if param == 0:
			redeploy.append(f'  {a:4}  skip redeploy (leave your people where they are)')
		elif param < MR:
			redeploy.append(f'  {a:4}  put {param} more ppl on EACH area you own')
		else:
			area = param - MR
			redeploy.append(f'  {a:4}  put 1 ppl on area {area:2}  {_area_str(territories, area)}')
	block('REDEPLOY your people at the end of the turn:', redeploy)

	other = []
	for a in legal:
		if a == 5*A+MR+DECK_SIZE:
			other.append(f'  {a:4}  DECLINE your active people (they stop conquering, keep scoring)')
		elif a == 5*A+MR+DECK_SIZE+1:
			other.append(f'  {a:4}  END your turn')
	block('OTHER:', other)

	return lines


