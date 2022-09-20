#!/usr/bin/env python3

import logging
import os
import coloredlogs
import argparse

from Coach import Coach
from santorini.SantoriniGame import SantoriniGame as Game
from santorini.NNet import NNetWrapper as nn
from utils import *
import subprocess
log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

def run(args):
	log.debug('Loading %s...', Game.__name__)
	g = Game()

	log.debug('Loading %s...', nn.__name__)
	nn_args = dict(
		lr=args.learn_rate,
		dropout=0.3,
		epochs=args.epochs,
		batch_size=args.batch_size,
		nn_version=args.nn_version,
		learn_rate=args.learn_rate,
		vl_weight=args.vl_weight,
		cyclic_lr=args.cyclic_lr,
		surprise_weight=args.surprise_weight,
		no_compression=args.no_compression,
	)
	nnet = nn(g, nn_args)

	if args.load_model:
		log.info('Loading checkpoint "%s"...', args.load_folder_file)
		nnet.load_checkpoint(os.path.dirname(args.load_folder_file), os.path.basename(args.load_folder_file))
		compare_settings(args)
	# else:
	# 	log.warning('Not loading a checkpoint!')

	log.debug('Loading the Coach...')
	c = Coach(g, nnet, args)

	if args.load_model:
		log.info("Loading 'trainExamples' from file...")
		c.loadTrainExamples()

	# Backup code used for this run
	subprocess.run(f'mkdir -p "{args.checkpoint}/"', shell=True)
	subprocess.run(f'cp *py santorini/*py "{args.checkpoint}/"', shell=True)
	subprocess.run(f'[ -f "{args.checkpoint}/settings.txt" ] && mv "{args.checkpoint}/settings.txt" "{args.checkpoint}/settings."`date +%s` ;   echo "{args}" > "{args.checkpoint}/settings.txt"', shell=True)

	log.debug('Starting the learning process 🎉')
	c.learn()

# Compare current settings and settings of checkpoints, display main differences
def compare_settings(args):
	settings_file = os.path.join(os.path.dirname(args.load_folder_file), 'settings.txt')

	# Load settings
	if not os.path.isfile(settings_file):
		return
	with open(settings_file,'r') as f:
		previous_args = f.read()
    
	# Compute differences on dict versions
	previous_args_dict, current_args_dict = vars(eval('argparse.'+previous_args)), vars(args)
	changed_keys = set([k for k in set(list(previous_args_dict.keys()) + list(current_args_dict.keys())) if previous_args_dict.get(k) != current_args_dict.get(k)])
	for key in ['load_folder_file', 'checkpoint', 'timeIters', 'numIters', 'arenaCompare', 'maxlenOfQueue', 'load_model']:
		changed_keys.discard(key)

	if changed_keys:
		log.info('Some option(s) changed compared to loaded checkpoint:')
		for k in changed_keys:
			print(f'{k}: {previous_args_dict.get(k)} --> {current_args_dict.get(k)}')

def profiling(args):
	import cProfile, pstats
	profiler = cProfile.Profile()
	args.numIters, args.numEps, args.epochs = 1, 1, 1 # warmup run
	run(args)

	print('\nstart profiling')
	args.numIters, args.numEps, args.epochs = 1, 30, 1
	# Core of the training
	profiler.enable()
	run(args)
	profiler.disable()

	# debrief
	profiler.dump_stats('execution.prof')
	# pstats.Stats(profiler).sort_stats('cumtime').print_stats(20)
	# print()
	# pstats.Stats(profiler).sort_stats('tottime').print_stats(10)
	print('check dumped stats in execution.prof')

def main():
	parser = argparse.ArgumentParser(description='tester')

	parser.add_argument('--numIters'        , '-n' , action='store', default=50   , type=int  , help='')
	parser.add_argument('--timeIters'       , '-t' , action='store', default=0.   , type=float, help='')
	parser.add_argument('--numEps'          , '-e' , action='store', default=500  , type=int  , help='Number of complete self-play games to simulate during a new iteration')
	parser.add_argument('--tempThreshold'   , '-T' , action='store', default=10   , type=int  , help='')
	parser.add_argument('--updateThreshold'        , action='store', default=0.60 , type=float, help='During arena playoff, new neural net will be accepted if threshold or more of games are won')
	# parser.add_argument('--maxlenOfQueue'   , '-q' , action='store', default=400000, type=int , help='Number of game examples to train the neural networks')
	parser.add_argument('--numMCTSSims'     , '-m' , action='store', default=1600 , type=int  , help='Number of moves for MCTS to simulate in FULL exploration')
	parser.add_argument('--ratio-fullMCTS'         , action='store', default=5    , type=int  , help='Ratio of MCTS sims between full and fast exploration')
	parser.add_argument('--prob-fullMCTS'          , action='store', default=0.25 , type=float, help='Probability to choose full MCTS exploration')
	parser.add_argument('--cpuct'           , '-c' , action='store', default=1.0  , type=float, help='')
	parser.add_argument('--dirichletAlpha'  , '-d' , action='store', default=0.2  , type=float, help='α=0.3 for chess, scaled in inverse proportion to the approximate number of legal moves in a typical position')    
	parser.add_argument('--numItersHistory' , '-i' , action='store', default=5   , type=int  , help='')

	parser.add_argument('--learn-rate'      , '-l' , action='store', default=0.0003, type=float, help='')
	parser.add_argument('--epochs'          , '-p' , action='store', default=2    , type=int  , help='')
	parser.add_argument('--batch-size'      , '-b' , action='store', default=32   , type=int  , help='')
	parser.add_argument('--nn-version'      , '-V' , action='store', default=1    , type=int  , help='Which architecture to choose')
	parser.add_argument('--vl-weight'       , '-v' , action='store', default=10.  , type=float, help='Weight for value loss')
	parser.add_argument('--forced-playouts' , '-F' , action='store_true', help='Enabled forced playouts')
	parser.add_argument('--cyclic-lr'       , '-Y' , action='store_true', help='Enable cyclic learning rate')
	parser.add_argument('--surprise-weight' , '-W' , action='store_true', help='Give more learning weights to surprising results')

	parser.add_argument('--no-compression'  , '-z' , action='store_true', help='Prevent using in-memory data compression (huge memory decrease and impact by only by ~1 second per 100k samples), useful for easier debugging')
	parser.add_argument('--no-mem-optim'    , '-Z' , action='store_true', help='Prevent cleaning MCTS tree of old moves during each game')
	parser.add_argument('--checkpoint'      , '-C' , action='store', default='./temp/', help='')
	parser.add_argument('--load-folder-file', '-L' , action='store', default=None     , help='')
	
	parser.add_argument('--profile'         , '-P' , action='store_true', help='profiler')
	
	args = parser.parse_args()
	args.arenaCompare = 30 if args.numEps < 500 else 50
	args.maxlenOfQueue = int(2.5e6/(1.2*args.numItersHistory)) # at most 2GB per process, with each example weighing 1.2kB
	if args.timeIters > 0:
		args.numIters = 1000

	args.load_model = (args.load_folder_file is not None)
	if args.profile:
		profiling(args)
	else:
		print(args)
		run(args)

if __name__ == "__main__":
	main()
