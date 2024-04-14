#!/usr/bin/env python3
import importlib
import itertools
import json
import multiprocessing
import os.path
import subprocess
from math import inf

import numpy as np

import Arena
from MCTS import MCTS
from utils import *
from GameSwitcher import import_game

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

game = None
_lock = multiprocessing.Lock()


def create_player(name, args):
	global game
	global NNet
	global players
	if game is None:
		Game, NNet, players, NUMBER_PLAYERS = import_game(args.game)
		game = Game()
	# all players
	if name == 'random':
		return players.RandomPlayer(game).play
	if name == 'greedy':
		return players.GreedyPlayer(game).play
	if name == 'human':
		return players.HumanPlayer(game).play

	# set default values but will be overloaded when loading checkpoint
	nn_args = dict(lr=None, dropout=0., epochs=None, batch_size=None, nn_version=-1)
	net = NNet(game, nn_args)
	cpt_dir, cpt_file = os.path.split(name)
	additional_keys = net.load_checkpoint(cpt_dir, cpt_file)

	cpuct = additional_keys.get('cpuct')
	cpuct = float(cpuct[0]) if isinstance(cpuct, list) else cpuct
	mcts_args = dotdict({
		'numMCTSSims'     : args.numMCTSSims if args.numMCTSSims else additional_keys.get('numMCTSSims', 100),
		'fpu'             : args.fpu if args.fpu else additional_keys.get('fpu', 0.),
		'universes'       : additional_keys.get('universes', 1),
		'cpuct'           : args.cpuct if args.cpuct else cpuct,
		'prob_fullMCTS'   : 1.,
		'forced_playouts' : False,
		'no_mem_optim'    : False,
	})
	mcts = MCTS(game, net, mcts_args)
	player = lambda x, n: np.argmax(mcts.getActionProb(x, temp=(0.5 if n <= 6 else 0.), force_full_search=True)[0])
	return player


def play(args):
	players = [p + '/best.pt' if os.path.isdir(p) else p for p in args.players]

	if not args.useray:
		print(players[0], 'vs', players[1])
	player1, player2 = create_player(players[0], args), create_player(players[1], args)
	human = 'human' in players
	arena = Arena.Arena(player1, player2, game, display=game.printBoard)
	result = arena.playGames(args.num_games, initial_state=args.state, verbose=args.display or human)

	if args.useray:
		##### Write results in a file
		directory = args.players[1] if os.path.isdir(args.players[1]) else os.path.dirname(args.players[1])
		score = result[1] + result[2] / 2.
		print('Writing score to ' + directory + '/score.txt:  ', score)
		with open(directory + '/score.txt', 'w') as f:
			f.write(f'{score}')
		#####

	return result


def play_age(args):
	players = subprocess.check_output(
		['find', args.compare, '-name', 'best.pt', '-mmin', '-' + str(args.compare_age * 60)])
	players = players.decode('utf-8').strip().split('\n')
	print(players)
	list_tasks = list(itertools.combinations(players, 2))
	plays(list_tasks, args)


def plays(list_tasks, args, callback_results=None):
	import math
	import time
	n = len(list_tasks)
	nb_tasks_per_thread = math.ceil(n / args.max_compare_threads)
	nb_threads = math.ceil(n / nb_tasks_per_thread)
	if nb_threads > 1:
		current_threads_list = subprocess.check_output(['ps', '-e', '-o', 'cmd']).decode('utf-8').split('\n')
		idx_thread = sum([1 for t in current_threads_list if 'pit.py' in t]) - 1
		if idx_thread == 0:
			print(f'\t{n} pits to do, splitted in {nb_tasks_per_thread} tasks * {nb_threads} threads')
		if idx_thread < nb_threads - 1:
			print(f'\tPlease call same script {nb_threads - 1 - idx_thread} time(s) more in other console')
		elif idx_thread >= nb_threads:
			print(f'I already have enough processes, exiting current one')
			exit()
	else:
		idx_thread = 0
		if n > 1:
			print(f'\t{n} pits to do')

	last_kbd_interrupt = 0.
	for (p1, p2) in list_tasks[idx_thread::nb_threads]:
		args.players = [p1, p2]
		try:
			game_results = play(args)

		except KeyboardInterrupt:
			now = time.time()
			if now - last_kbd_interrupt < 10:
				exit(0)
			last_kbd_interrupt = now
			print('Skipping this pit (hit CRTL-C once more to stop all)')
		else:
			if callback_results:
				callback_results(p1, p2, game_results, args)


def load_rating(player_file):
	import glicko2
	basename = os.path.splitext(os.path.basename(player_file))[0]
	rating_file = os.path.dirname(player_file) + '/rating' + ('' if basename == 'best' else '_' + basename) + '.json'
	if not os.path.exists(rating_file):
		return glicko2.Player()
	r_dict = json.load(open(rating_file, 'r'))
	return glicko2.Player(rating=r_dict['rating'], rd=r_dict['rd'], vol=r_dict['vol'])


def write_rating(rating_object, player_file):
	basename = os.path.splitext(os.path.basename(player_file))[0]
	rating_file = os.path.dirname(player_file) + '/rating' + ('' if basename == 'best' else '_' + basename) + '.json'
	rating_dict = {'rating': rating_object.rating, 'rd': rating_object.rd, 'vol': rating_object.vol}
	json.dump(rating_dict, open(rating_file, 'w'))


def update_ratings(p1, p2, game_results, args):
	oneWon, twoWon, draws = game_results
	with _lock:
		player1, player2 = load_rating(p1), load_rating(p2)
		p1r, p1rd = player1.rating, player1.rd
		p2r, p2rd = player2.rating, player2.rd
		n = oneWon + twoWon + draws
		player1.update_player([p2r] * n, [p2rd] * n, [1] * oneWon + [0.5] * draws + [0] * twoWon)
		player2.update_player([p1r] * n, [p1rd] * n, [1] * twoWon + [0.5] * draws + [0] * oneWon)
		write_rating(player1, args.players[0])
		write_rating(player2, args.players[1])
		# for p, pname in [(player1, p1), (player2, p2)]:
		# 	print(f'{pname[-20:].rjust(20)} rating={int(p.rating)}±{int(p.rd)}, vol={p.vol:.3e}')

def play_several_files(args):
	players = args.players[:]  # Copy, because it will be overwritten by plays()
	list_tasks = []
	if args.reference:
		if args.useray:
			list_tasks += list(itertools.product(args.reference, args.players))
		else:
			list_tasks += list(itertools.product(args.players, args.reference))
	if not args.vs_ref_only:
		list_tasks += list(itertools.combinations(args.players, 2))

	if args.ratings:
		plays(list_tasks, args, callback_results=update_ratings)
		for p in players:
			r = load_rating(p)
			name = os.path.basename(os.path.dirname(p)) + (
				'' if os.path.basename(p) == 'best.pt' else (' - ' + os.path.basename(p)))
			print(f'{name[-20:].ljust(20)} rating={int(r.rating)}±{int(r.rd)}, vol={r.vol:.3e}')
	else:
		worst_games = {}

		def show_worst_game(p1, p2, result, _):
			worst_games[p1] = min(result[0], worst_games.get(p1, inf))
			worst_games[p2] = min(result[1], worst_games.get(p2, inf))

		plays(list_tasks, args, callback_results=show_worst_game)
		if len(players) > 3:
			for name, worst_game in worst_games.items():
				print(f'{name:<40}: {worst_game}')


def profiling(args):
	import cProfile, pstats

	args.num_games = 4
	profiler = cProfile.Profile()
	print('\nstart profiling')
	profiler.enable()

	# Core of the training
	print(play(args))

	# debrief
	profiler.disable()
	profiler.dump_stats('execution.prof')
	pstats.Stats(profiler).sort_stats('cumtime').print_stats(20)
	print()
	pstats.Stats(profiler).sort_stats('tottime').print_stats(10)


def main():
	import argparse
	parser = argparse.ArgumentParser(description='tester')  

	parser.add_argument('--num-games'          , '-n' , action='store', default=30   , type=int  , help='')
	parser.add_argument('--profile'                   , action='store_true', help='enable profiling')
	parser.add_argument('--display'                   , action='store_true', help='display')
	parser.add_argument('--state'              , '-s' , action='store', default="", type=str, help='State to load')

	parser.add_argument('--numMCTSSims'        , '-m' , action='store', default=None, type=int  , help='Number of games moves for MCTS to simulate.')
	parser.add_argument('--cpuct'              , '-c' , action='store', default=None, type=float, help='cpuct value')
	parser.add_argument('--fpu'                , '-f' , action='store', default=None, type=float, help='Value for FPU (first play urgency)')

	parser.add_argument('game'                        , action='store', default='splendor', help='The name of the game to play')
	parser.add_argument('players'                     , metavar='player', nargs='*', help='list of players to test (either file, or "human" or "random")')
	parser.add_argument('--reference'          , '-r' , metavar='ref'   , nargs='*', help='list of reference players')
	parser.add_argument('--vs-ref-only'        , '-z' , action='store_true', help='Use this option to prevent games between players, only players vs references')
	parser.add_argument('--ratings'            , '-R' , action='store_true', help='Compute ratings based in games results and write ratings on disk')
	parser.add_argument('--useray'                    , action='store_true', help='Mode for "ray", disable some messages')

	parser.add_argument('--compare'            , '-C' , action='store', default='../results', help='Compare all best.pt located in the specified folders')
	parser.add_argument('--compare-age'        , '-A' , action='store', default=None        , help='Maximum age (in hour) of best.pt to be compared', type=int)
	parser.add_argument('--max-compare-threads', '-T' , action='store', default=1           , help='No of threads to run comparison on', type=int)

	args = parser.parse_args()
	
	if args.profile:
		profiling(args)
	elif args.compare_age:
		play_age(args)
	elif args.reference or len(args.players) > 2:
		play_several_files(args)
	elif len(args.players) == 2:
		play(args)
	else:
		raise Exception('Please specify a player (ai folder, random, greedy or human)')


if __name__ == "__main__":
	main()
