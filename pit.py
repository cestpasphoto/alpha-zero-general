import Arena
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from utils import *
import os.path
from os import stat

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

game = OthelloGame(6)

def create_player(name, args):
	# all players
	if name == 'random':
		return RandomPlayer(game).play
	if name == 'greedy':
		return GreedyPlayer(game).play
	if name == 'human':
		return HumanPlayer(game).play

	# set default values but will be overloaded when loading checkpoint
	nn_args = dict(lr=None, dropout=0., epochs=None, batch_size=None, num_channels=0)
	net = NNet(game, nn_args)
	cpt_dir, cpt_file = os.path.split(name)
	net.load_checkpoint(cpt_dir, cpt_file)
	mcts = MCTS(game, net, dotdict({'numMCTSSims': args.numMCTSSims, 'cpuct': args.cpuct}))
	player = lambda x: np.argmax(mcts.getActionProb(x, temp=0))
	return player

def play(args):
	if None in [args.player1, args.player2]:
		raise Exception('Please specify a player (ai folder, random, greedy or human)')
	if args.singlethread:
		import torch
		torch.set_num_threads(1)
	if os.path.isdir(args.player2):
		args.player2 += '/best.pt'
	p2_name = os.path.basename(os.path.dirname(args.player2))

	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress tensorflow warning about cuda not available
	results = []

	for p1 in args.player1:
		if p1.endswith('/'):
			p1 = p1[:-1]
		if os.path.isdir(p1):
			p1_list = []
			writer = SummaryWriter(log_dir='logs/'+os.path.basename(p1), purge_step=0)
			for root, dirs, files in os.walk(p1):
				for file in files:
					if file.startswith('checkpoint') and file.endswith('.pt'): 
						p1_list.append(p1+'/'+file)
			p1_list = sorted(p1_list, key=lambda name: os.path.basename(name).split('.')[0][11:])
		else:
			writer = None
			p1_list = [p1]

		for i, single_p1 in enumerate(p1_list):
			print(single_p1, 'vs', args.player2)
			player1, player2 = create_player(single_p1, args), create_player(args.player2, args)
			human = 'human' in [single_p1, args.player2]
			arena = Arena.Arena(player1, player2, game, display=display)
			result = arena.playGames(args.num_games, verbose=args.display or human)
			results.append(result)
			if writer is not None:
				net = NNet(game)
				cpt_dir, cpt_file = os.path.split(single_p1)
				net.load_checkpoint(cpt_dir, cpt_file)
				runtime = net.cumulated_uptime + net.begin_time
				num = int(cpt_file.split('.')[0][11:])
				score = (result[0]+result[2]/2) / args.num_games
				if i == 0:
					writer.add_scalar('perf_vs_'+p2_name, 0, 0, net.begin_time)
				writer.add_scalar('perf_vs_'+p2_name, score, num, runtime)
	return results

def display(numpy_board):
	print_board(Board(2, numpy_board))

def profiling(args):
	import cProfile, pstats

	args.num_games, args.singlethread = 5, True
	np.setbufsize(1024)
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
	parser.add_argument('--singlethread', '-s', action='store_true', help='single thread')
	parser.add_argument('--profile'           , action='store_true', help='enable profiling')
	parser.add_argument('--display'           , action='store_true', help='display')

	parser.add_argument('--numMCTSSims', '-m' , action='store', default=5   , type=int  , help='Number of games moves for MCTS to simulate.')
	parser.add_argument('--cpuct'      , '-c' , action='store', default=1.0  , type=float, help='')

	parser.add_argument('--player1'    , '-p' , action='store', default=None , nargs='*', help='P1: either file or human, greedy, random')
	parser.add_argument('--player2'    , '-P' , action='store', default=None , help='P2: either file or human, greedy, random')

	args = parser.parse_args()
	
	if args.profile:
		profiling(args)
	else:
		play(args)

if __name__ == "__main__":
	main()
