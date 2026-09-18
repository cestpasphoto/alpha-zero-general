import numpy as np
from colorama import Style, Fore

try:
	from .CatanConstants import *
except ImportError:
	from CatanConstants import *

############################# COLOURS AND GLYPHS ##############################

player_color = [Fore.CYAN, Fore.YELLOW, Fore.MAGENTA, Fore.GREEN]
hex_color = [Fore.WHITE, Fore.RED, Fore.GREEN, Fore.BLACK, Fore.YELLOW, Fore.CYAN]
hex_char = ['DES', 'BRI', 'LUM', 'ORE', 'GRA', 'WOO']       # desert / hill / forest / mountain / field / pasture
res_char = ['brick', 'lumber', 'ore', 'grain', 'wool']
res_short = ['B', 'L', 'O', 'G', 'W']
dev_char = ['knight', 'victory point', 'road building', 'monopoly', 'year of plenty']
port_char = ['', '2:1 brick', '2:1 lumber', '2:1 ore', '2:1 grain', '2:1 wool', '3:1 any']
port_short = ['', '2B', '2L', '2O', '2G', '2W', '3*']
phase_char = ['setup settlement', 'setup road', 'roll', 'discard', 'move robber', 'main',
              'road building', 'trade offer', 'trade answer', 'trade accept']

############################# ACTION TO STRING ################################

def move_to_str(move, player=0, short=False):
	p = f'P{player} '
	if A_ROAD <= move < A_ROAD + N_EDGES:
		e = move - A_ROAD
		return f'{p}road on edge {e}' if not short else f'rd{e}'
	if A_SETTLEMENT <= move < A_SETTLEMENT + N_VERTICES:
		v = move - A_SETTLEMENT
		return f'{p}settlement on vertex {v}' if not short else f'st{v}'
	if A_CITY <= move < A_CITY + N_VERTICES:
		v = move - A_CITY
		return f'{p}city on vertex {v}' if not short else f'ci{v}'
	if move == A_BUY_DEV:
		return f'{p}buys a development card' if not short else 'buyDev'
	if move == A_PLAY_DEV + 0:
		return f'{p}plays a knight' if not short else 'knight'
	if move == A_PLAY_DEV + 1:
		return f'{p}plays road building' if not short else 'roadBld'
	if A_ROBBER <= move < A_ROBBER + N_HEXES * N_PLAYERS:
		h, t = divmod(move - A_ROBBER, N_PLAYERS)
		who = 'nobody' if t == 0 else f'P{(player + t) % N_PLAYERS}'
		return f'{p}moves the robber to hex {h} and robs {who}' if not short else f'rob{h}>{who}'
	if move == A_ROLL:
		return f'{p}rolls the dice' if not short else 'roll'
	if A_MONOPOLY <= move < A_MONOPOLY + N_RESOURCES:
		r = move - A_MONOPOLY
		return f'{p}plays monopoly on {res_char[r]}' if not short else f'mono{res_short[r]}'
	if A_YEAR_OF_PLENTY <= move < A_YEAR_OF_PLENTY + 15:
		r1, r2 = YOP_PAIRS[move - A_YEAR_OF_PLENTY]
		return (f'{p}plays year of plenty, takes {res_char[r1]} + {res_char[r2]}' if not short
		        else f'yop{res_short[r1]}{res_short[r2]}')
	if A_BANK_TRADE <= move < A_BANK_TRADE + 20:
		give, k = divmod(move - A_BANK_TRADE, 4)
		get = k if k < give else k + 1
		return (f'{p}trades {res_char[give]} for {res_char[get]}' if not short
		        else f'tr{res_short[give]}>{res_short[get]}')
	if A_DISCARD <= move < A_DISCARD + N_RESOURCES:
		r = move - A_DISCARD
		return f'{p}discards one {res_char[r]}' if not short else f'dis{res_short[r]}'
	if move == A_END_TURN:
		return f'{p}ends turn' if not short else 'end'
	if A_TRADE_RECV <= move < A_TRADE_RECV + N_TRADE_SETS:
		s = set_to_str(TRADE_SETS[move - A_TRADE_RECV])
		return f'{p}asks for {s}' if not short else f'ask{s}'
	if A_TRADE_GIVE <= move < A_TRADE_GIVE + N_TRADE_SETS:
		s = set_to_str(TRADE_SETS[move - A_TRADE_GIVE])
		return f'{p}offers {s} in exchange' if not short else f'give{s}'
	if move == A_TRADE_OK:
		return f'{p}accepts the offer' if not short else 'OK'
	if move == A_TRADE_NO:
		return f'{p}declines the offer' if not short else 'NO'
	if A_TRADE_ACCEPT <= move < A_TRADE_ACCEPT + N_PLAYERS:
		t = move - A_TRADE_ACCEPT
		if t == 0:
			return f'{p}refuses every counter-offer' if not short else 'refAll'
		who = f'P{(player + t) % N_PLAYERS}'
		return f'{p}takes the counter-offer from {who}' if not short else f'take{who}'
	return f'{p}unknown action {move}' if not short else f'?{move}'


def set_to_str(counts):
	"""A trade multiset as e.g. '2L+1G'. Shared by move_to_str and the board dump,
	so a hand and an offer are always read the same way."""
	return '+'.join(f'{int(n)}{res_short[r]}' for r, n in enumerate(counts) if n) or '-'


############################# PRINT GAME ######################################
#
# The canvas is built from the layout coordinates exported by CatanConstants, so
# it cannot drift from the topology: a hex centre is (2q+r, 3r) in doubled
# pointy-top coordinates, a vertex the mean over its 3 hexes, an edge the mean
# over its 2. Everything is placed on one integer grid.

_ROW = lambda y: int(round(y * 2)) + 16          # 0 .. 32
_COL = lambda x: int(round(x * 6)) + 32          # 0 .. 64


def _blit(canvas, row, col, text, color=''):
	if row < 0 or row >= len(canvas):
		return
	col -= len(text) // 2
	for i, ch in enumerate(text):
		if 0 <= col + i < len(canvas[row]):
			canvas[row][col + i] = (color + ch + Style.RESET_ALL) if color else ch


def _board_lines(board):
	canvas = [[' '] * 66 for _ in range(33)]
	vy, vx = VERTEX_POS[:, 1] / 3., VERTEX_POS[:, 0] / 3.
	ey, ex = EDGE_POS[:, 1] / 2., EDGE_POS[:, 0] / 2.

	# hexes: resource, token, robber
	for h in range(N_HEXES):
		t, tok = board.hexes[h, H_TYPE], board.hexes[h, H_TOKEN]
		label = hex_char[t] if t == HEX_DESERT else f'{hex_char[t]}{TOKEN_VALUES[tok]}'
		if board.hexes[h, H_ROBBER]:
			label = '*' + label          # the robber sits on the hex label, not on a row of its own
		_blit(canvas, _ROW(HEX_POS[h, 1]), _COL(HEX_POS[h, 0]), label,
		      Fore.RED if board.hexes[h, H_ROBBER] else hex_color[t])

	# edges: a road, oriented from the two endpoints
	for e in range(N_EDGES):
		owner = board.vertices[EDGE_TO_VERTEX[e, 0], V_EDGE0 + EDGE_SLOT[e, 0]]
		a, b = EDGE_TO_VERTEX[e]
		if vy[a] == vy[b]:
			glyph = '---'
		else:
			hi, lo = (a, b) if vy[a] < vy[b] else (b, a)
			glyph = '\\' if vx[hi] < vx[lo] else '/'
		_blit(canvas, _ROW(ey[e]), _COL(ex[e]), glyph,
		      player_color[owner - 1] if owner else Style.DIM)

	# vertices: settlement, city, or a free port slot
	for v in range(N_VERTICES):
		o, b = board.vertices[v, V_OWNER], board.vertices[v, V_BUILDING]
		if o:
			_blit(canvas, _ROW(vy[v]), _COL(vx[v]), 'o' if b == 1 else 'O', player_color[o - 1])
		elif board.vertices[v, V_PORT] != PORT_NONE:
			_blit(canvas, _ROW(vy[v]), _COL(vx[v]), port_short[board.vertices[v, V_PORT]], Fore.BLUE)
		else:
			_blit(canvas, _ROW(vy[v]), _COL(vx[v]), '.', Style.DIM)

	return [''.join(r).rstrip() for r in canvas if ''.join(r).strip()]


def _player_lines(board):
	lines = []
	for p in range(board.num_players):
		a, b, c = board.players[4*p], board.players[4*p + 1], board.players[4*p + 2]
		d = board.players[4*p + 3]
		res = ' '.join(f'{res_short[r]}{a[PA_RESOURCES + r]}' for r in range(N_RESOURCES))
		dev = ' '.join(f'{dev_char[k][:3]}{a[PA_DEV_PLAYABLE + k] + b[PB_DEV_NEW + k]}'
		               for k in range(N_DEV_TYPES) if a[PA_DEV_PLAYABLE + k] + b[PB_DEV_NEW + k])
		ports = ' '.join(port_short[i + 1] for i in range(6) if c[PC_PORTS + i])
		bonus = ('+road' if b[PB_HAS_ROAD] else '') + ('+army' if b[PB_HAS_ARMY] else '')
		lines.append(f'{player_color[p]}P{p}{Style.RESET_ALL} '
		             f'{c[PC_VP_PUBLIC] + c[PC_VP_DEV]:2d}vp  {res}  |  road{b[PB_ROAD_LENGTH]} '
		             f'kn{b[PB_KNIGHTS]}{bonus}  |  left {b[PB_SETTLEMENTS_LEFT]}s '
		             f'{b[PB_CITIES_LEFT]}c {b[PB_ROADS_LEFT]}r  |  {dev}  {ports}')
		# a standing offer is public, so it belongs on the visible board dump
		st = int(d[PD_TRADE_STATUS])
		if st == TRADE_COMPOSING:
			lines.append(f'      {Fore.BLUE}asking {set_to_str(d[PD_TRADE_RECV:PD_TRADE_RECV + N_RESOURCES])}'
			             f', offer pending{Style.RESET_ALL}')
		elif st == TRADE_OFFERED:
			lines.append(f'      {Fore.BLUE}offers {set_to_str(d[PD_TRADE_GIVE:PD_TRADE_GIVE + N_RESOURCES])}'
			             f' -> wants {set_to_str(d[PD_TRADE_RECV:PD_TRADE_RECV + N_RESOURCES])}{Style.RESET_ALL}')
		elif st == TRADE_REFUSED:
			lines.append(f'      {Style.DIM}declined the offer{Style.RESET_ALL}')
	return lines


def print_board(board):
	g = board.globals_
	rnd = int(g[1, GB_ROUND_HI]) * 100 + int(g[1, GB_ROUND_LO])
	dice = g[0, GA_DICE]
	print()
	traded = ' (trade spent)' if g[1, GB_PLAYER_TRADE_DONE] else ''
	print(f'turn {rnd}  |  phase {phase_char[g[0, GA_PHASE]]}{traded}  |  '
	      f'dice {dice if dice else "-"}  |  turn of P{g[1, GB_TURN_PLAYER]}  |  '
	      f'bank {" ".join(f"{res_short[r]}{g[0, GA_BANK + r]}" for r in range(N_RESOURCES))}  |  '
	      f'dev deck {sum(g[0, GA_DEV_DECK + k] for k in range(N_DEV_TYPES))}')
	for line in _board_lines(board):
		print(line)
	for line in _player_lines(board):
		print(line)
