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
import glicko2

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
	nn_args = dict(lr=None, dropout=0., epochs=None, batch_size=None, dense2d=[1], dense1d=[1])
	net = NNet(game, nn_args)
	cpt_dir, cpt_file = os.path.split(name)
	net.load_checkpoint(cpt_dir, cpt_file)
	mcts = MCTS(game, net, dotdict({'numMCTSSims': args.numMCTSSims, 'cpuct': args.cpuct}))
	player = lambda x: np.argmax(mcts.getActionProb(x, temp=0))
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

def compute_elo(args):
	def print_ratings(data):
		ratings = [(model, data[model]['rating'], data[model]['rd']) for model in data]
		ratings = sorted(ratings, key=lambda x: x[1])
		for r in ratings:
			print(f'{round(r[1]):>4} ±{round(r[2]):<3} {r[0]}')

	import json
	filename = 'glicko.json'
	data = json.load(open(filename, 'r'))
	if len(data.keys()) == 0:
		rnd = glicko2.Player()
		data['random'] = {'rating': rnd.rating, 'rd': rnd.rd, 'vol': rnd.vol}

	p1_name = args.player1
	p1_results, other_results = [], {}
	for opponent in data.keys():
		args.player2 = opponent
		other_results[opponent] = []
		results = play(args)
		for _ in range(results[0]): # WIN
			p1_results.append( (data[opponent]['rating'], data[opponent]['rd'], 1) )
			other_results[opponent].append(0)
		for _ in range(results[1]): # LOSS
			p1_results.append( (data[opponent]['rating'], data[opponent]['rd'], 0) )
			other_results[opponent].append(1)
		for _ in range(results[2]): # DRAW
			p1_results.append( (data[opponent]['rating'], data[opponent]['rd'], 0.5) )
			other_results[opponent].append(0.5)

	# Changes on p1
	p1 = glicko2.Player()
	p1.update_player([x[0] for x in p1_results], [x[1] for x in p1_results], [x[2] for x in p1_results])
	data[p1_name] = {'rating': p1.rating, 'rd': p1.rd, 'vol': p1.vol}
	# Changes on others
	for opponent in [d for d in data.keys() if d != p1_name]:
		p2 = glicko2.Player(data[opponent]['rating'], data[opponent]['rd'], data[opponent]['vol'])
		p2_results = other_results[opponent]
		p2.update_player([p1.rating]*len(p2_results), [p1.rd]*len(p2_results), p2_results)
		data[opponent] = {'rating': p2.rating, 'rd': p2.rd, 'vol': p2.vol}

	print_ratings(data)
	with open(filename, 'w') as file:
		json.dump(data, file, indent=2)

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
	parser.add_argument('--profile'           , action='store_true', help='enable profiling')
	parser.add_argument('--display'           , action='store_true', help='display')
	parser.add_argument('--elo'         , '-e', action='store_true', help='compute elo-like ranking (glicko-2)')

	parser.add_argument('--numMCTSSims', '-m' , action='store', default=5   , type=int  , help='Number of games moves for MCTS to simulate.')
	parser.add_argument('--cpuct'      , '-c' , action='store', default=1.0  , type=float, help='')

	parser.add_argument('--player1'    , '-p' , action='store', default=None , help='P1: either file or human, greedy, random')
	parser.add_argument('--player2'    , '-P' , action='store', default=None , help='P2: either file or human, greedy, random')

	args = parser.parse_args()
	
	if args.profile:
		profiling(args)
	elif args.elo:
		compute_elo(args)
	else:
		play(args)

if __name__ == "__main__":
	main()
