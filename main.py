#!../venv/bin/python3

import logging
import os
import coloredlogs

from Coach import Coach
from splendor.SplendorGame import SplendorGame as Game
from splendor.NNet import NNetWrapper as nn
from utils import *
import subprocess
log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.


def run(args):
	log.debug('Loading %s...', Game.__name__)
	g = Game(2)

	log.debug('Loading %s...', nn.__name__)
	nn_args = dict(lr=args.learn_rate, dropout=0.3, epochs=args.epochs, batch_size=args.batch_size, dense2d=args.dense2d, dense1d=args.dense1d)
	nnet = nn(g, nn_args)

	if args.load_model:
		log.info('Loading checkpoint "%s"...', args.load_folder_file)
		nnet.load_checkpoint(os.path.dirname(args.load_folder_file), os.path.basename(args.load_folder_file), ongoing_experiment=False)
	# else:
	# 	log.warning('Not loading a checkpoint!')

	log.debug('Loading the Coach...')
	c = Coach(g, nnet, args)

	if args.load_model:
		log.info("Loading 'trainExamples' from file...")
		c.loadTrainExamples()

	subprocess.run(f'mkdir -p "{args.checkpoint}/"', shell=True)
	subprocess.run(f'cp *py splendor/*py "{args.checkpoint}/"', shell=True)
	subprocess.run(f'echo "{args}" >> "{args.checkpoint}/main.py"', shell=True)

	log.debug('Starting the learning process 🎉')
	c.learn()

def profiling(args):
	import cProfile, pstats
	profiler = cProfile.Profile()
	print('\nstart profiling')
	args.numIters, args.numEps, args.epochs = 1, 20, 1 # also, add a "return" just after self-play in Coach.py

	# Core of the training
	profiler.enable()
	run(args)
	profiler.disable()

	# debrief
	profiler.dump_stats('execution.prof')
	pstats.Stats(profiler).sort_stats('cumtime').print_stats(20)
	print()
	pstats.Stats(profiler).sort_stats('tottime').print_stats(10)

def game_test():
	from splendor.SplendorGame import test_game
	test_game(generate_test=False)

def main():
	import argparse
	parser = argparse.ArgumentParser(description='tester')	

#	cpuct = 1.0 ... ou plus (pas d'impact sur le temps)
#   rollout = joue sur la perf et le temps...
#	learn_rate = 0.001 ? ou bien 0.02 puis diviser à chaque raté ?
	parser.add_argument('--numIters'        , '-N' , action='store', default=50   , type=int  , help='')
	parser.add_argument('--timeIters'       , '-T' , action='store', default=0.   , type=float, help='')
	parser.add_argument('--numEps'          , '-s' , action='store', default=100   , type=int  , help='Number of complete self-play games to simulate during a new iteration')
	parser.add_argument('--tempThreshold'   , '-t' , action='store', default=15    , type=int  , help='')
	parser.add_argument('--updateThreshold' , '-u' , action='store', default=0.55 , type=float, help='During arena playoff, new neural net will be accepted if threshold or more of games are won')
	# parser.add_argument('--maxlenOfQueue'   , '-q' , action='store', default=400000, type=int , help='Number of game examples to train the neural networks')
	parser.add_argument('--numMCTSSims'     , '-m' , action='store', default=5    , type=int  , help='Number of games moves for MCTS to simulate.')
	parser.add_argument('--cpuct'           , '-c' , action='store', default=1.0   , type=float, help='')
	parser.add_argument('--dirichletAlpha'  , '-a' , action='store', default=0.1  , type=float, help='α=0.3 for chess, scaled in inverse proportion to the approximate number of legal moves in a typical position')    
	parser.add_argument('--numItersForTrainExamplesHistory', '-n', action='store', default=5, type=int, help='')

	parser.add_argument('--learn-rate'      , '-l' , action='store', default=0.001, type=float, help='')
	parser.add_argument('--epochs'          , '-e' , action='store', default=5    , type=int  , help='')
	parser.add_argument('--batch-size'      , '-b' , action='store', default=128   , type=int  , help='')
	
	parser.add_argument('--checkpoint'      , '-C' , action='store', default='./temp/', help='')
	parser.add_argument('--load-folder-file', '-L' , action='store', default=None     , help='')
	
	parser.add_argument('--profile'         , '-P' , action='store_true', help='profiler')
	parser.add_argument('--test'                   , action='store_true', help='test logic')
	parser.add_argument('--dense2d'         , '-X' , action='store', default=[256, 256], nargs='*', type=int, help='(CNN archi) nb of linear layers BEFORE flattening')
	parser.add_argument('--dense1d'         , '-x' , action='store', default=[512, 256, 256], nargs='*', type=int, help='(CNN archi) nb of linear layers AFTER flattening')
	
	args = parser.parse_args()
	args.arenaCompare = 30
	args.maxlenOfQueue = int(2e6/(1.1*args.numItersForTrainExamplesHistory)) # at most 2GB per process, with each example weighing 1.1kB
	if args.timeIters > 0:
		args.numIters = 1000

	args.load_model = (args.load_folder_file is not None)
	if args.profile:
		profiling(args)
	elif args.test:
		game_test()
	else:
		print(args)
		run(args)

if __name__ == "__main__":
	main()
