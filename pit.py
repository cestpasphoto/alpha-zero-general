#!../venv/bin/python3

import Arena
from MCTS import MCTS
from splendor.SplendorPlayers import *
from splendor.SplendorGame import SplendorGame as Game
from splendor.SplendorLogic import print_board
from splendor.SplendorLogicNumba import Board
from splendor.NNet import NNetWrapper as NNet

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
		game = Game(2)
	# all players
	if name == 'random':
		return RandomPlayer(game).play
	if name == 'greedy':
		return GreedyPlayer(game).play
	if name == 'human':
		return HumanPlayer(game).play

	# set default values but will be overloaded when loading checkpoint
	nn_args = dict(lr=None, dropout=0., epochs=None, batch_size=None, nn_version=-1, save_optim_state=False)
	net = NNet(game, nn_args)
	cpt_dir, cpt_file = os.path.split(name)
	additional_keys = net.load_checkpoint(cpt_dir, cpt_file)
	mcts_args = dotdict({
		'numMCTSSims'     : args.numMCTSSims if args.numMCTSSims else additional_keys.get('numMCTSSims', 100),
		'cpuct'           : args.cpuct       if args.cpuct       else additional_keys.get('cpuct'      , 1.0),
		'prob_fullMCTS'   : 1.,
		'forced_playouts' : False,
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
	players = subprocess.check_output(['find', args.compare, '-name', 'best.pt', '-mmin', '-'+str(args.compare_age*60)])
	players = players.decode('utf-8').strip().split('\n')
	n = len(players)

	nb_iterations = math.ceil(n/args.compare_threads)
	target_nb_threads = math.ceil(n/nb_iterations)
	current_threads_list = subprocess.check_output(['ps', '-e', '-o', 'cmd']).decode('utf-8').split('\n')
	idx_thread = sum([1 for t in current_threads_list if 'pit.py' in t]) - 1
	if idx_thread == 0:
		print(players)
		print(f'\t{n} models, will need {nb_iterations} * {n//2} iterations * {target_nb_threads} threads')
	if idx_thread < target_nb_threads-1:
		print(f'\tPlease call same script {target_nb_threads-1-idx_thread} time(s) more in other console')
	elif idx_thread >= target_nb_threads:
		print(f'I already have enough processes, exiting current one')
		exit()

	for p1 in range(idx_thread, n, target_nb_threads):
		args.player1 = players[p1]
		for p2_delta in range(1, 1+n//2):
			args.player2 = players[ (p1 + p2_delta)%n ]
			play(args)

def display(numpy_board):
	board = Board(2)
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

	parser.add_argument('--num-games'      , '-n' , action='store', default=30   , type=int  , help='')
	parser.add_argument('--profile'           , action='store_true', help='enable profiling')
	parser.add_argument('--display'           , action='store_true', help='display')

	parser.add_argument('--numMCTSSims'    , '-m' , action='store', default=None  , type=int  , help='Number of games moves for MCTS to simulate.')
	parser.add_argument('--cpuct'          , '-c' , action='store', default=None  , type=float, help='')

	parser.add_argument('--player1'        , '-p' , action='store', default=None , help='P1: either file or human, greedy, random')
	parser.add_argument('--player2'        , '-P' , action='store', default=None , help='P2: either file or human, greedy, random')

	parser.add_argument('--compare'        , '-C' , action='store', default='../results', help='Compare all best.pt located in the specified folders')
	parser.add_argument('--compare-age'    , '-A' , action='store', default=None        , help='Maximum age (in hour) of best.pt to be compared', type=int)
	parser.add_argument('--compare-threads', '-T' , action='store', default=6           , help='No of threads to run comparison on', type=int)

	args = parser.parse_args()
	
	if args.profile:
		profiling(args)
	elif args.compare:
		plays(args)
	else:
		play(args)

if __name__ == "__main__":
	main()
