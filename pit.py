#!../venv/bin/python3

import Arena
from MCTS import MCTS
from splendor.SplendorPlayers import *
from splendor.SplendorGame import SplendorGame as Game
from splendor.SplendorLogic import Board, print_board
from splendor.NNet import NNetWrapper as NNet

import numpy as np
from utils import *
import os.path
from os import stat

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

game = Game(2)

def create_player(name, args):
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
	net.load_checkpoint(cpt_dir, cpt_file)
	mcts = MCTS(game, net, dotdict({'numMCTSSims': args.numMCTSSims, 'prob_fullMCTS': 1., 'cpuct': args.cpuct}))
	player = lambda x: np.argmax(mcts.getActionProb(x, temp=0)[0])
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

	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress tensorflow warning about cuda not available
	results = []

	print(args.player1, 'vs', args.player2)
	player1, player2 = create_player(args.player1, args), create_player(args.player2, args)
	human = 'human' in [args.player1, args.player2]
	arena = Arena.Arena(player1, player2, game, display=display)
	result = arena.playGames(args.num_games, verbose=args.display or human)
	return result

def display(numpy_board):
	print_board(Board(2, numpy_board))

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

	parser.add_argument('--num-games'  , '-n' , action='store', default=30   , type=int  , help='')
	parser.add_argument('--profile'           , action='store_true', help='enable profiling')
	parser.add_argument('--display'           , action='store_true', help='display')

	parser.add_argument('--numMCTSSims', '-m' , action='store', default=100  , type=int  , help='Number of games moves for MCTS to simulate.')
	parser.add_argument('--cpuct'      , '-c' , action='store', default=1.0  , type=float, help='')

	parser.add_argument('--player1'    , '-p' , action='store', default=None , help='P1: either file or human, greedy, random')
	parser.add_argument('--player2'    , '-P' , action='store', default=None , help='P2: either file or human, greedy, random')

	args = parser.parse_args()
	
	if args.profile:
		profiling(args)
	else:
		play(args)

if __name__ == "__main__":
	main()
