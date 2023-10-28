import random
import json
from smallworld.SmallworldLogicNumba import *
from smallworld.SmallworldGame import SmallworldGame
from os import listdir, path, mkdir, _exit

b = Board(NUMBER_PLAYERS)
game = SmallworldGame()
p, it = 0, 0


def compute_valids_all_game():
	global game, b, p
	valids = game.getValidMoves(b, p)
	if len(valids) == 0:
		breakpoint()
	return valids

def do_action_game(action):
	global game, b, p
	b, p = game.getNextState(b, p, action)

def play_one_move_game(dump_directory=None):
	global game, b, p, it

	if dump_directory:
		backup_state_before = b.copy()
		current_player = p

	valids = compute_valids_all_game()
	action = random.choices(range(len(valids)), valids)[0]

	if 123 <= action <= 128 and p == 1:
		# Force P1 to choose amazons
		amazon_index = [i for i in range(DECK_SIZE) if game.board.visible_deck[i,1] == AMAZON]
		if len(amazon_index):
			if valids[amazon_index[0] + 123]:
				action = amazon_index[0] + 123

	do_action_game(action)
	
	if dump_directory:
		backup_state_after = b.copy()
		data_dict = {
			'before' : backup_state_before.tolist(),
			'player' : current_player,
			'valids' : valids.tolist(),
			'action' : action,
			'after'  : backup_state_after.tolist(),
		}
		pretty_write_json(data_dict, dump_directory + f'/dump{it:03}.json')

	print()
	print(f'P{current_player} {move_to_str(action)}, {action=} {it=}')
	print_board(game.board)
	it += 1

	return action

def play_full_game(dump_directory=None):
	global game, b, p, it
	b = game.getInitBoard()

	if dump_directory and not path.exists(dump_directory):
		mkdir(dump_directory)

	while not game.getGameEnded(b, p).any():
		play_one_move_game(dump_directory)

def compare_move_to_reference(dump_file, print_at_begin=False):
	global p, b, game
	with open(dump_file, 'r') as f:
		reference = json.load(f)

	def do_if_difference(action=None):
		if action:
			print(f'{p=} {move_to_str(action)} {action=}')
		else:
			print(f'{p=}')
		print_board(game.board)
		breakpoint()

	b = np.array(reference['before'], dtype=np.int8)
	_ = game.getRound(b) # Set game to reference
	p = reference['player']

	if print_at_begin:
		print_board(game.board)

	valids = compute_valids_all_game()
	if valids.tolist() != reference['valids']:
		print('error in valids')
		do_if_difference()
		return

	action = reference['action']
	do_action_game(action)

	board_state = b.copy().tolist()
	for i in range(len(board_state)):
		# Do not compare invisible_deck, nor last item of visible deck (random)
		if 123 <= action <= 128 and (i == NB_AREAS+3*NUMBER_PLAYERS+DECK_SIZE-1 or i >= NB_AREAS+4*NUMBER_PLAYERS+DECK_SIZE):
			continue
		# Do not compare prerun dice info for berserk
		if board_state[i][2] == BERSERK and NB_AREAS <= i < NB_AREAS+3*NUMBER_PLAYERS:
			if (board_state[i][:4] != reference['after'][i][:4]) or \
			   (board_state[i][4] & 2**6) != (reference['after'][i][4] & 2**6):
				print(f'error in prerun data after action {action}, row {i}')
				do_if_difference(action)
		elif board_state[i] != reference['after'][i]:
			print(f'error in state after action {action}, row {i}')
			do_if_difference(action)	

def compare_to_references(dump_directory):
	dump_files = [dump_directory+f for f in listdir(dump_directory) if f.startswith('dump')]
	for dump_file in sorted(dump_files):
		print('\r'+dump_file+' ', end='')
		compare_move_to_reference(dump_file)

###############################################################################

def pretty_write_json(data_dict, filename):
	with open(filename, 'w') as f:
		f.write('{\n')
		for k, v in data_dict.items():
			f.write(json.dumps(k) + ': ' + json.dumps(v) + ',\n')
		f.write('"zfake": 0\n}\n')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--game', action='store_true')
parser.add_argument('--tests', action='store_true')
parser.add_argument('-t')
args = parser.parse_args()

main_directory = './smallworld/dumps/'

if args.game:
	used_ids = [int(f[3:6]) for f in listdir(main_directory) if f.startswith('set')]
	new_id = max(used_ids)+1 if len(used_ids) else 0
	dump_directory = f'{main_directory}/set{new_id:03}/'
	play_full_game(dump_directory)

elif args.tests:
	for dump_directory in [f'{main_directory}/{f}/' for f in sorted(listdir(main_directory)) if f.startswith('set')]:
		compare_to_references(dump_directory)
	print()

elif args.t:
	compare_move_to_reference(args.t, print_at_begin=True)

else:
	print('No action')