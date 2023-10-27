import random
import json
from smallworld.SmallworldLogicNumba import *
from smallworld.SmallworldGame import SmallworldGame
from os import listdir, path, mkdir, _exit

b = Board(NUMBER_PLAYERS)
game = SmallworldGame()
p, it = 0, 0

########################  GAME VERSION  #######################################

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
	valids = compute_valids_all_game()
	action = random.choices(range(len(valids)), valids)[0]
	do_action_game(action)
	print()
	print(f'P{p} {action=}, {it=}')
	print_board(game.board)
	it += 1

def play_full_game():
	global game, b, p, it
	b = game.getInitBoard()
	while not game.getGameEnded(b, p).any():
		play_one_move_game()

########################  "DIRECT LOGIC" VERSION ##############################

def compute_valids_all(p, do_print):
	valids_attack    = b._valids_attack(p)
	valids_specialppl= b._valids_special_actionppl(p)
	valids_abandon   = b._valids_abandon(p)
	valids_redeploy  = b._valids_redeploy(p)
	valids_specialpwr= b._valids_special_actionpwr(p)
	valids_choose    = b._valids_choose_ppl(p)
	valid_decline    = b._valid_decline(p)
	valid_end        = b._valid_end(p)
	if do_print:
		print_valids(p, valids_attack, valids_specialppl, valids_abandon, valids_redeploy, valids_specialpwr, valids_choose, valid_decline, valid_end)

	valids_all = \
		[(i, 'attack'    ) for i, v in enumerate(valids_attack) if v] +\
		[(i, 'specialppl') for i, v in enumerate(valids_specialppl) if v] +\
		[(i, 'abandon'   ) for i, v in enumerate(valids_abandon) if v] +\
		[(i, 'redeploy'  ) for i, v in enumerate(valids_redeploy) if v] +\
		[(i, 'specialpwr') for i, v in enumerate(valids_specialpwr) if v] +\
		[(i, 'choose'    ) for i, v in enumerate(valids_choose) if v] +\
		[(i, 'decline'   ) for i, v in enumerate([valid_decline]) if v] +\
		[(i, 'end'       ) for i, v in enumerate([valid_end]) if v]

	return valids_all

def do_action(p, area, action):
	if   action == 'attack':
		print(f'Attacking area {area}')
		b._do_attack(p, area)
	elif action == 'specialppl':
		print(f'Special movePPL on area {area}')
		b._do_special_actionppl(p, area)
	elif action == 'abandon':
		print(f'Abandonning area {area}')
		b._do_abandon(p, area)
	elif action == 'redeployeach':
		param = area
		print(f'Redeploy {area}ppl on each area')
		b._do_redeploy(p, param)
	elif action == 'redeploy1':
		param = area+MAX_REDEPLOY
		print(f'Redeploy on area {area}')
		b._do_redeploy(p, param)
	elif action == 'specialpwr':
		print(f'Special movePWR on area {area}')
		b._do_special_actionpwr(p, area)
	elif action == 'choose':
		print(f'Choose ppl #{area}')
		b._do_choose_ppl(p, area)	
	elif action == 'decline':
		print(f'Decline current ppl')
		b._do_decline(p)
	elif action == 'end':
		print('end turn')
		b._do_end(p)
	else:
		breakpoint()

def play_one_turn(dump_directory=None):
	global p, it
	p = (b.status[:, 3] >= 0).nonzero()[0].item()
	print('  ' + '='*20 + f'  P{p} now plays  ' + '='*20)

	if not path.exists(dump_directory):
		mkdir(dump_directory)

	while b.status[p, 3] >= 0:
		valids_all = compute_valids_all(p, do_print=True)
		weights = [4 if t == 'attack' else 4 if t == 'decline' else 0.5 if t == 'redeploy' else 0.1 if t == 'end' else 1.0 for _,t in valids_all]
		if len(valids_all) == 0:
			print('No possible action')
			breakpoint()

		if dump_directory:
			backup_state_before = b.get_state().copy()

		area, action = random.choices(valids_all, weights)[0]

		# Chose a "redeploy on each" action if possible
		if action == 'redeploy':
			valids_on_each = [area_ for (area_, action_) in valids_all if (action_ == 'redeploy' and area_ < MAX_REDEPLOY)]
			if len(valids_on_each):
				area, action = max(valids_on_each), 'redeployeach'
			else:
				area, action = area-MAX_REDEPLOY, 'redeploy1'
		# try:
		# 	do_action(p, area, action)
		# except Exception as e:
		# 	stop = True
		# 	print(e)
		# 	breakpoint()
		# else:
		# 	stop = False
		# 	print_board(b)
		# 	print()


		do_action(p, area, action)
		print_board(b)
		print()
		stop = False


		if dump_directory:
			backup_state_after = b.get_state().copy()
			with open(dump_directory + f'/dump{it:03}.json', 'w') as f:
				dump_data = {
					'before'     : backup_state_before.tolist(),
					'player'     : p,
					'valids_all' : valids_all,
					'area_action': [area, action],
					'after'      : backup_state_after.tolist(),
				}
				f.write('{\n')
				for k, v in dump_data.items():
					f.write(json.dumps(k) + ': ' + json.dumps(v) + ',\n')
				f.write('"zfake": 0\n}\n')

		it += 1

		if stop:
			print('Exit with error')
			_exit(-1)

	return (p+1)%NUMBER_PLAYERS

def run_test(dump_file):
	global p
	with open(dump_file, 'r') as f:
		dump_data = json.load(f)

	ref_before = np.array(dump_data['before'], dtype=np.int8)
	b.copy_state(ref_before, True)
	p = dump_data['player']

	valids_all = compute_valids_all(p, do_print=True)
	ref_valids_all = [tuple(x) for x in dump_data['valids_all']]
	if valids_all != ref_valids_all:
		print('error in valids')
		breakpoint()

	ref_area, ref_action = dump_data['area_action']
	do_action(p, ref_area, ref_action)

	board_state = b.get_state().tolist()
	for i in range(len(board_state)):
		# Do not compare invisible_deck, nor last item of visible deck (random)
		if ref_action == 'choose' and (i == NB_AREAS+3*NUMBER_PLAYERS+DECK_SIZE-1 or i >= NB_AREAS+4*NUMBER_PLAYERS+DECK_SIZE):
			continue
		# Do not compare prerun dice info for berserk
		if board_state[i][2] == BERSERK and NB_AREAS <= i < NB_AREAS+3*NUMBER_PLAYERS:
			if (board_state[i][:4] != dump_data['after'][i][:4]) or \
			   (board_state[i][4] & 2**6) != (dump_data['after'][i][4] & 2**6):
				print(f'error in after, row {i}')
				breakpoint()
		elif board_state[i] != dump_data['after'][i]:
			print(f'error in after, row {i}')
			breakpoint()

def run_tests(dump_directory):
	dump_files = [dump_directory+f for f in listdir(dump_directory) if f.startswith('dump')]
	for dump_file in sorted(dump_files):
		run_test(dump_file)

###############################################################################

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--game', action='store_true')
parser.add_argument('--create', action='store_true')
parser.add_argument('--tests', action='store_true')
parser.add_argument('--profile', action='store_true')
args = parser.parse_args()

if args.game:
	play_full_game()
elif args.create:
	# Create a new testset
	used_ids = [int(f[3:6]) for f in listdir('./dumps/') if f.startswith('set')]
	new_id = max(used_ids)+1 if len(used_ids) else 0
	dump_directory = f'./dumps/set{new_id:03}/'
	print(f'Dump dir: {dump_directory}')

	print_board(b)
	print()
	while not b.check_end_game(p).any():
		p = play_one_turn(dump_directory=dump_directory)

	print(f'The end: {b.check_end_game(p)}')

	run_tests(dump_directory)

elif args.tests:
	directories = ['./dumps/'+f+'/' for f in listdir('./dumps/') if f.startswith('validated')]
	for directory in directories:
		run_tests(directory)

elif args.profile:
	import cProfile, pstats
	profiler = cProfile.Profile()

	directories = ['./dumps/'+f+'/' for f in listdir('./dumps/') if f.startswith('validated')]
	for directory in directories:
		run_tests(directory)

	profiler.enable()
	for _ in range(100):
		for directory in directories:
			run_tests(directory)
	profiler.disable()

	profiler.dump_stats('execution.prof')
	from pstats import Stats, SortKey
	p = Stats('execution.prof')
	p.strip_dirs().sort_stats('cumtime').print_stats(20)
	print()
	p.strip_dirs().sort_stats('tottime').print_stats(10)
	breakpoint()


else:
	print('No action')