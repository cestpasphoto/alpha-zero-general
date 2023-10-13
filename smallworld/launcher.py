import random
import json
from SmallworldLogicNumba import *
from os import remove, listdir, path

b = Board(NUMBER_PLAYERS)
p, it = 0, 0

def compute_valids_all(p, do_print):
	valids_attack    = b._valids_attack(player=p)
	valids_specialppl= b._valids_special_actionppl(player=p)
	valids_abandon   = b._valids_abandon(player=p)
	valids_redeploy  = b._valids_redeploy(player=p)
	valids_specialpwr= b._valids_special_actionpwr(player=p)
	valids_choose    = b._valids_choose_ppl(player=p)
	valid_decline    = b._valid_decline(player=p)
	valid_end        = b._valid_end(player=p)
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
		b._do_attack(player=p, area=area)
	elif action == 'specialppl':
		print(f'Special movePPL on area {area}')
		b._do_special_actionppl(player=p, area=area)
	elif action == 'abandon':
		print(f'Abandonning area {area}')
		b._do_abandon(player=p, area=area)
	elif action == 'redeployeach':
		param = area
		print(f'Redeploy {area}ppl on each area')
		b._do_redeploy(player=p, param=param)
	elif action == 'redeploy1':
		param = area+MAX_REDEPLOY
		print(f'Redeploy on area {area}')
		b._do_redeploy(player=p, param=param)
	elif action == 'specialpwr':
		print(f'Special movePWR on area {area}')
		b._do_special_actionpwr(player=p, area=area)
	elif action == 'choose':
		print(f'Choose ppl #{area}')
		b._do_choose_ppl(player=p, index=area)	
	elif action == 'decline':
		print(f'Decline current ppl')
		b._do_decline(player=p)
	elif action == 'end':
		print('end turn')
		b._do_end(player=p)
	else:
		breakpoint()

def play_one_turn(dump_directory=None):
	global p, it
	p = (b.status[:, 3] >= 0).nonzero()[0].item()
	print('  ' + '='*20 + f'  P{p} now plays  ' + '='*20)

	while b.status[p, 3] >= 0:
		valids_all = compute_valids_all(p, do_print=True)
		weights = [3 if t == 'attack' else 0.5 if t == 'redeploy' else 0.1 if t == 'end' else 1.0 for _,t in valids_all]
		if len(valids_all) == 0:
			print('No possible action')
			breakpoint()

		if dump_directory:
			backup_state_before = b.get_state().copy()

		area, action = random.choices(valids_all, weights)[0]

		# Chose a "redeploy on each" action if possible
		if action == 'redeploy':
			valids_on_each = [area_ for (area_, action_) in valids_all if (action_ == 'redeploy' and area_ < MAX_REDEPLOY)]
			if any(valids_on_each):
				area, action = max(valids_on_each), 'redeployeach'
			else:
				area, action = area-MAX_REDEPLOY, 'redeploy1'
		try:
			do_action(p, area, action)
		except:
			pass
		else:
			print_board(b)
			print()

		if dump_directory:
			backup_state_after = b.get_state().copy()
			if it == 0:
				# Remove previous files
				_ = [remove(path.join(dump_directory, f)) for f in listdir(dump_directory) if f.startswith('dump')]
			# And dump
			with open(dump_directory + f'/dump{it:03}.json', 'w') as f:
				dump_data = {
					'before'     : backup_state_before.tolist(),
					'player'     : p,
					'valids_all' : valids_all,
					'area_action': [area, action],
					'after'      : backup_state_after.tolist(),
				}
				f.write('{')
				for k, v in dump_data.items():
					f.write(json.dumps(k) + ': ' + json.dumps(v) + ',\n')
				f.write('"fake": 0\n}\n')

			it += 1

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
		if ref_action == 'choose' and i in [len(board_state) - 1, len(board_state) - 4]:
			continue
		if board_state[i] != dump_data['after'][i]:
			print(f'error in after, row {i}')
			breakpoint()

###############################################################################

print_board(b)
print()
while not b.check_end_game(p).any():
	p = play_one_turn(dump_directory='./dumps/')

print(f'The end: {b.check_end_game(p)}')

run_test('dumps/dump000.json')