import random
import json
from os import listdir, path, mkdir, _exit


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
	action = random.choice([i for i,v in enumerate(valids) if v])

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
			reference_board_before = Board(NUMBER_PLAYERS)
			reference_board_before.copy_state(np.array(reference['before'], dtype=np.int8), True)
			print_board(reference_board_before)
			print(f'P{reference["player"]} {move_to_str(action)} {action=}')
			reference_board_after = Board(NUMBER_PLAYERS)
			reference_board_after.copy_state(np.array(reference['after'], dtype=np.int8), True)
			print_board(reference_board_after)
		else:
			print(f'P{reference["player"]}')
			print_board(game.board)
		breakpoint()

	b = np.array(reference['before'], dtype=np.int8)
	_ = game.getRound(b) # Set game to reference
	p = reference['player']

	if print_at_begin:
		print_board(game.board)

	valids = compute_valids_all_game()
	if valids.tolist() != reference['valids']:
		print('⚠️ error in valids')
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
				print(f'⚠️ error in prerun data on row {i}: before={reference["before"][i]} ref={reference["after"][i]} new={board_state[i]} ')
				do_if_difference(action)
		elif board_state[i] != reference['after'][i]:
			print(f'⚠️ error in state row {i}: before={reference["before"][i]} ref={reference["after"][i]} new={board_state[i]} ')
			do_if_difference(action)	

def compare_to_references(dump_directory):
	dump_files = [dump_directory+f for f in listdir(dump_directory) if f.startswith('dump')]
	for dump_file in sorted(dump_files):
		print('\r'+dump_file+' ', end='')
		compare_move_to_reference(dump_file)

def detect_infinite_loops():
	global game, b, p, it
	b = game.getInitBoard()

	actions_list = []
	while not game.getGameEnded(b, p).any():
		action = play_one_move_game()
		actions_list.append(action)

		# Check if one of the prev actions are still possible
		valids = compute_valids_all_game()
		for i in range(2, 4):
			if len(actions_list) < i:
				continue
			if all([92<=a<=122 for a in actions_list[-i:]]): # If ongoing redeploy, that's normal if previous action is the same
				continue
			previous_action = actions_list[-i]
			if valids[previous_action]:
				print(f'Warning action {previous_action} "{move_to_str(previous_action)}" is still possible ({-i})')
				breakpoint()


###############################################################################

def pretty_write_json(data_dict, filename):
	with open(filename, 'w') as f:
		f.write('{\n')
		for k, v in data_dict.items():
			f.write(json.dumps(k) + ': ' + json.dumps(v) + ',\n')
		f.write('"zfake": 0\n}\n')

import argparse
from GameSwitcher import *

parser = argparse.ArgumentParser()
parser.add_argument('game' , action='store', default='splendor', help='The name of the game to play')
parser.add_argument('--play', action='store_true')
parser.add_argument('--tests', action='store_true')
parser.add_argument('-t')
parser.add_argument('--loops', action='store_true')
args = parser.parse_args()
game, nnet_unused, players_unused, NUMBER_PLAYERS = import_game(args.game)
Board = import_logicnumba(args.game)
b = Board(NUMBER_PLAYERS)
p, it = 0, 0

main_directory = './' + args.game + '/dumps/'

if args.play:
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

elif args.loops:
	detect_infinite_loops()

else:
	print('No action')