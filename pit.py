#!/usr/bin/env python3

import Arena
from MCTS import MCTS
from santorini.SantoriniPlayers import *
from santorini.SantoriniGame import SantoriniGame as Game
from santorini.SantoriniGame import NUMBER_PLAYERS
from santorini.SantoriniDisplay import print_board
from santorini.SantoriniLogicNumba import Board
from santorini.NNet import NNetWrapper as NNet

import numpy as np
from utils import *
import os.path
from os import stat

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

game = None

def create_player(name, args):
	global game
	if game is None:
		game = Game()
	# all players
	if name == 'random':
		return RandomPlayer(game).play
	if name == 'greedy':
		return GreedyPlayer(game).play
	if name == 'human':
		return HumanPlayer(game).play

	# set default values but will be overloaded when loading checkpoint
	nn_args = dict(lr=None, dropout=0., epochs=None, batch_size=None, nn_version=-1)
	net = NNet(game, nn_args)
	cpt_dir, cpt_file = os.path.split(name)
	additional_keys = net.load_checkpoint(cpt_dir, cpt_file)
	mcts_args = dotdict({
		'numMCTSSims'     : args.numMCTSSims if args.numMCTSSims else additional_keys.get('numMCTSSims', 100),
		'cpuct'           : args.cpuct       if args.cpuct       else additional_keys.get('cpuct'      , 1.0),
		'prob_fullMCTS'   : 1.,
		'forced_playouts' : False,
		'no_mem_optim'    : False,
	})
	mcts = MCTS(game, net, mcts_args)
	player = lambda x: np.argmax(mcts.getActionProb(x, temp=0, force_full_search=True)[0])
	return player

def play(args):
	if None in [args.player1, args.player2]:
		raise Exception('Please specify a player (ai folder, random, greedy or human)')
	if os.path.isdir(args.player2):
		args.player2 += '/best.pt'
	p2_name = os.path.basename(os.path.dirname(args.player2))
	if os.path.isdir(args.player1):
		args.player1 += '/best.pt'
	p1_name = os.path.basename(os.path.dirname(args.player1))

	results = []
	print(args.player1, 'vs', args.player2)
	player1, player2 = create_player(args.player1, args), create_player(args.player2, args)
	human = 'human' in [args.player1, args.player2]
	arena = Arena.Arena(player1, player2, game, display=display)
	result = arena.playGames(args.num_games, verbose=args.display or human)
	return result

def plays(args):
	import subprocess
	import math
	import itertools
	import time
	players = subprocess.check_output(['find', args.compare, '-name', 'best.pt', '-mmin', '-'+str(args.compare_age*60)])
	players = players.decode('utf-8').strip().split('\n')
	list_tasks = list(itertools.combinations(players, 2))
	n = len(list_tasks)

	nb_tasks_per_thread = math.ceil(n/args.max_compare_threads)
	nb_threads = math.ceil(n/nb_tasks_per_thread)
	current_threads_list = subprocess.check_output(['ps', '-e', '-o', 'cmd']).decode('utf-8').split('\n')
	idx_thread = sum([1 for t in current_threads_list if 'pit.py' in t]) - 1
	if idx_thread == 0:
		print(players)
		print(f'\t{n} pits to do, splitted in {nb_tasks_per_thread} tasks * {nb_threads} threads')
	if idx_thread < nb_threads-1:
		print(f'\tPlease call same script {nb_threads-1-idx_thread} time(s) more in other console')
	elif idx_thread >= nb_threads:
		print(f'I already have enough processes, exiting current one')
		exit()

	last_kbd_interrupt = 0.
	for (p1, p2) in list_tasks[idx_thread::nb_threads]:
		args.player1, args.player2 = p1, p2
		try:
			play(args)
		except KeyboardInterrupt:
			now = time.time()
			if now - last_kbd_interrupt < 10:
				exit(0)
			last_kbd_interrupt = now
			print('Skipping this pit (hit CRTL-C once more to stop all)')

def display(numpy_board):
	board = Board(NUMBER_PLAYERS)
	board.copy_state(numpy_board, False)
	print_board(board)

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
	parser.add_argument('--convert'                   , action='store', default='', help='Old network to transform to new format')
	parser.add_argument('--display'                   , action='store_true', help='display')

	parser.add_argument('--numMCTSSims'        , '-m' , action='store', default=None  , type=int  , help='Number of games moves for MCTS to simulate.')
	parser.add_argument('--cpuct'              , '-c' , action='store', default=None  , type=float, help='')

	parser.add_argument('--player1'            , '-p' , action='store', default=None        , help='P1: either file or human, greedy, random')
	parser.add_argument('--player2'            , '-P' , action='store', default=None        , help='P2: either file or human, greedy, random')

	parser.add_argument('--compare'            , '-C' , action='store', default='../results', help='Compare all best.pt located in the specified folders')
	parser.add_argument('--compare-age'        , '-A' , action='store', default=None        , help='Maximum age (in hour) of best.pt to be compared', type=int)
	parser.add_argument('--max-compare-threads', '-T' , action='store', default=6           , help='No of threads to run comparison on', type=int)

	args = parser.parse_args()
	
	if args.profile:
		profiling(args)
	elif args.convert:
		to_new_format(args.convert)
	elif args.compare_age:
		plays(args)
	else:
		play(args)

if __name__ == "__main__":
	main()
