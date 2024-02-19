import logging
import os
import sys
from collections import deque
import pickle
import zlib
from tqdm import tqdm, trange
from queue import SimpleQueue
from threading import Thread, Lock
from time import sleep

from random import shuffle
import numpy as np

from Arena import Arena
from MCTS import MCTS

log = logging.getLogger(__name__)

class Coach():
	"""
	This class executes the self-play + learning. It uses the functions defined
	in Game and NeuralNet. args are specified in main.py.
	"""

	def __init__(self, game, nnet, args):
		self.game = game
		self.nnet = nnet
		self.pnet = self.nnet.__class__(self.game, self.nnet.args)  # the competitor network
		self.args = args
		self.mcts = MCTS(self.game, self.nnet, self.args, dirichlet_noise=(self.args.dirichletAlpha!=0))
		self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
		self.skipFirstSelfPlay = nnet.requestKnowledgeTransfer  # can be overriden in loadTrainExamples()
		self.consecutive_failures = 0
		self.nb_threads = self.args.parallel_inferences

	def executeEpisode(self, my_mcts=None, my_game=None):
		"""
		This function executes one episode of self-play, starting with player 1.
		As the game is played, each turn is added as a training example to
		trainExamples. The game is played till the game ends. After the game
		ends, the outcome of the game is used to assign values to each example
		in trainExamples.

		It uses a temp=1 if episodeStep < tempThreshold, and thereafter
		uses temp=0.

		Returns:
			trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
						   pi is the MCTS informed policy vector, v is +1 if
						   the player eventually won the game, else -1.
		"""
		if my_mcts is None:
			my_mcts = self.mcts
		if my_game is None:
			my_game = self.game
		trainExamples = []
		board = my_game.getInitBoard()
		curPlayer = 0
		episodeStep = 0

		while True:
			episodeStep += 1
			canonicalBoard = my_game.getCanonicalForm(board, curPlayer)
			pi, q, is_full_search = my_mcts.getActionProb(canonicalBoard, temp=1.)
			action = random_pick(pi, temperature=2 if episodeStep < self.args.tempThreshold else self.args.temperature[1])

			if is_full_search:
				valids = my_game.getValidMoves(canonicalBoard, 0)
				sym = my_game.getSymmetries(canonicalBoard, pi, valids)
				for b, p, v in sym:
					trainExamples.append([b, p, curPlayer, v, q])

			board, curPlayer = my_game.getNextState(board, curPlayer, action)

			r = my_game.getGameEnded(board, curPlayer)
			if r.any():
				final_scores = [my_game.getScore(board, p) for p in range(my_game.num_players)]
				trainExamples = [(
					x[0],                                # board
					x[1],                                # policy
					np.roll(r, -x[2]),                   # winner
					x[3],                                # valids
					x[4],                                # Q estimates
				) for x in trainExamples]

				return trainExamples if self.args.no_compression else [zlib.compress(pickle.dumps(x), level=1) for x in trainExamples]

	def executeEpisodes_batch(self, i_thread, shared_memory, locks):
		# Execute an episode in a thread until need to evaluate NN
		# then unlock next threads, etc until batch of inferences to do is full
		# then server runs inferences on batch.
		# Each thread loops until receiving a signal to stop
		locks[i_thread].acquire()
		batch_info = (i_thread, i_thread+self.nb_threads, shared_memory, locks)
		while shared_memory[-1] == 0: # Signal 0 means to continue computing
			my_game = self.game.__class__()
			my_game.getInitBoard()
			my_mcts = MCTS(my_game, self.nnet, self.args, dirichlet_noise=(self.args.dirichletAlpha!=0), batch_info=batch_info)
			episode = self.executeEpisode(my_mcts, my_game)
			self.examplesQueue.put(episode)

		while shared_memory[-1] == 1: # We received signal 1, wait for other threads to complete
			locks[i_thread+1].release()
			locks[i_thread].acquire()
		locks[i_thread+1].release()

	def executeEpisodes(self):
		iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
		if self.nb_threads == 1:
			for _ in trange(self.args.numEps, desc="Self Play", ncols=120):
				iterationTrainExamples += self.executeEpisode()
				self.MCTS = MCTS(self.game, self.nnet, self.args, dirichlet_noise=(self.args.dirichletAlpha!=0))
				if len(iterationTrainExamples) == self.args.maxlenOfQueue:
					log.warning(f'saturation of elements in iterationTrainExamples, think about decreasing numEps or increasing maxlenOfQueue')
					break
		else:
			# N slots for NN inputs, N slots for NN ouputs, 1 slot for signaling
			# signal: 0 = compute, 1 = stop after current episode, 2 = stop
			shared_memory = [None] * (2*self.nb_threads) + [0]
			# list of Locks: "0;n-1" are MCTSs and "n" is the batch NN processor
			locks = [Lock() for _ in range(self.nb_threads+1)]

			self.examplesQueue = SimpleQueue()
			[l.acquire() for l in locks]
			threads_list = [Thread(target=self.executeEpisodes_batch, args=(i_thread, shared_memory, locks)) for i_thread in range(self.nb_threads)]
			threads_list.append(Thread(target=self.nnet.predict_server, args=(self.nb_threads, shared_memory, locks)))
			[t.start() for t in threads_list]

			progress = tqdm(total=self.args.numEps, desc="Self Play", ncols=120, smoothing=0.1, disable=None)
			nb_examples, max_nb_episodes = 0, self.args.numEps
			while True:
				sleep(1)
				for _ in range(self.examplesQueue.qsize()):
					iterationTrainExamples += self.examplesQueue.get_nowait()
					nb_examples += 1
					progress.update()
				# Check if we have collected enough samples
				if nb_examples >= self.args.numEps - self.nb_threads:
					if nb_examples >= max_nb_episodes:
						shared_memory[-1] = 2 # send signal 2 = all threads can be stopped
						break
					elif shared_memory[-1] == 0:
						max_nb_episodes = nb_examples + self.nb_threads
						progress.total = max_nb_episodes
						shared_memory[-1] = 1 # send signal 1 = threads can stop after their current episode
			[t.join() for t in threads_list]
			progress.close()

		MCTS.reset_all_search_trees()
		return iterationTrainExamples

	def learn(self):
		"""
		Performs numIters iterations with numEps episodes of self-play in each
		iteration. After every iteration, it retrains neural network with
		examples in trainExamples (which has a maximum length of maxlenofQueue).
		It then pits the new neural network against the old one and accepts it
		only if it wins >= updateThreshold fraction of games.
		"""

		for i in range(1, self.args.numIters + 1):
			# examples of the iteration
			if not self.skipFirstSelfPlay or i > 1:
				iterationTrainExamples = self.executeEpisodes()
				if len(iterationTrainExamples) == self.args.maxlenOfQueue:
					log.warning(f'saturation of elements in iterationTrainExamples, think about decreasing numEps or increasing maxlenOfQueue')

				# save the iteration examples to the history 
				self.trainExamplesHistory.append(iterationTrainExamples)

				# Check average number of valid moves, and compare to Dirichlet
				nb_valid_moves = [sum(pickle.loads(zlib.decompress(x))[3]) for x in iterationTrainExamples]
				avg_valid_moves = sum(nb_valid_moves) / len(nb_valid_moves)
				if self.args.dirichletAlpha > 0 and not (1/1.5 < self.args.dirichletAlpha / (10/avg_valid_moves) < 1.5):
					print(f'There are about {avg_valid_moves:.1f} valid moves per state, so I advise to set dirichlet to {10/avg_valid_moves:.1f} instead')

			if self.args.profile:
				return

			if len(self.trainExamplesHistory) > self.args.numItersHistory:
				self.trainExamplesHistory.pop(0)
			# backup history to a file
			self.saveTrainExamples()
			# shuffle examples before training
			trainExamples = []
			for e in self.trainExamplesHistory:
				trainExamples.extend(e)
			shuffle(trainExamples)

			# training new network, keeping a copy of the old one
			self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pt', additional_keys=vars(self.args))
			self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pt')
			pmcts = MCTS(self.game, self.pnet, self.args)

			self.nnet.train(trainExamples)
			nmcts = MCTS(self.game, self.nnet, self.args)

			# log.info('PITTING AGAINST PREVIOUS VERSION')
			arena = Arena(lambda x, n: np.argmax(nmcts.getActionProb(x, temp=(0.5 if n <= 6 else 0.), force_full_search=True)[0]),
						  lambda x, n: np.argmax(pmcts.getActionProb(x, temp=(0.5 if n <= 6 else 0.), force_full_search=True)[0]), self.game)
			nwins, pwins, draws = arena.playGames(self.args.arenaCompare)

			if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
				self.consecutive_failures += 1
				log.info(f'Iter #{i} - new vs previous: {nwins}-{pwins}  ({draws} draws) --> REJECTED ({self.consecutive_failures})')
				if self.consecutive_failures >= self.args.stop_after_N_fail and i < self.args.numIters:
					log.error('Exceeded threshold number of consecutive fails, stopping process')
					exit()
				self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pt')
			else:
				log.info(f'Iter #{i} - new vs previous: {nwins}-{pwins}  ({draws} draws) --> ACCEPTED')
				self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i), additional_keys=vars(self.args))
				self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pt', additional_keys=vars(self.args))
				self.consecutive_failures = 0

	def getCheckpointFile(self, iteration):
		return 'checkpoint_' + str(iteration) + '.pt'

	def saveTrainExamples(self):
		folder = self.args.checkpoint
		if not os.path.exists(folder):
			os.makedirs(folder)
		filename = os.path.join(folder, "checkpoint.examples")
		with open(filename, "wb") as f:
			pickle.dump(self.trainExamplesHistory, f)

	def loadTrainExamples(self):
		modelFile = self.args.load_folder_file
		examplesFile = os.path.dirname(modelFile) + "/checkpoint.examples"
		if not os.path.isfile(examplesFile):
			log.warning(f'File "{examplesFile}" with trainExamples not found!')
			if not self.args.useray:
				r = input("Continue? [y|n]")
				if r != "y":
					sys.exit()
			return
	
		log.info("File with trainExamples found. Loading it...")
		with open(examplesFile, "rb") as f:
			self.trainExamplesHistory = pickle.load(f)
		
		# Harmonize compression use in loaded examples
		if type(self.trainExamplesHistory[0][0]) is tuple and not self.args.no_compression:
			for i in range(len(self.trainExamplesHistory)):
				for j in range(len(self.trainExamplesHistory[i])):                    
					self.trainExamplesHistory[i][j] = zlib.compress(pickle.dumps(self.trainExamplesHistory[i][j]), level=1)
		elif type(self.trainExamplesHistory[0][0]) is not tuple and self.args.no_compression:
			for i in range(len(self.trainExamplesHistory)): 
				for j in range(len(self.trainExamplesHistory[i])):
					self.trainExamplesHistory[i][j] = pickle.loads(zlib.decompress(self.trainExamplesHistory[i][j]))
		log.info('Loading done!')

		# cleaning
		if len(self.trainExamplesHistory) > self.args.numItersHistory:
			self.trainExamplesHistory = self.trainExamplesHistory[-self.args.numItersHistory:]
			log.info('Reduced history in loaded examples')
		for history in self.trainExamplesHistory:
			if len(history) > self.args.maxlenOfQueue:
				for _ in range(len(history), self.args.maxlenOfQueue, -1):
					history.pop()
				log.info('Reduced nb of items in one history of loaded examples')


def applyTemperatureAndNormalize(probs, temperature):
	if temperature == 0:
		bests = np.array(np.argwhere(probs == np.max(probs))).flatten()
		result = [0] * len(probs)
		result[np.random.choice(bests)] = 1
	else:
		result = [x ** (1. / temperature) for x in probs]
		result_sum = float(sum(result))
		result = [x / result_sum for x in result]
	return result

def random_pick(probs, temperature=1.):
	probs_with_temp = applyTemperatureAndNormalize(probs, temperature)
	pick = np.random.choice(len(probs_with_temp), p=probs_with_temp)
	return pick

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description='Examples loader')
	parser.add_argument('input', metavar='example filename', nargs='*'                 , help='list of examples to load (.examples files)')
	parser.add_argument('--output'    , '-o', action='store', default='../results/new' , help='Prefix for output files')
	parser.add_argument('--binarize'  , '-b', action='store_true', help='Transform policy into binary one')
	args = parser.parse_args()

	training, testing = [], []
	for filename in args.input:
		print(f'Loading {filename}...')
		with open(filename, "rb") as f:
			new_input = pickle.load(f)
			print(f'size = {[len(x) for x in new_input]}, total = {sum([len(x) for x in new_input])}')
			training += new_input[:-1]
			testing += [list(x)[::8] for x in new_input[-1:]] # Remove symmetries

	# for filename in args.input:
	#     print(f'Loading {filename}...')
	#     with open(filename, "rb") as f:
	#         new_input = pickle.load(f)
	#         print(f'size = {[len(x) for x in new_input]}, total = {sum([len(x) for x in new_input])}')
	#         training += new_input[-3:]
	# testing = [list(training[-1])[::8]]
	# training = training[:-1]
	
	if args.binarize:
		print('Binarizing policy...')
		for t in [training, testing]:
			for i in range(len(t)):
				print(i, end=' ')
				for j in range(len(t[i])):
					data = pickle.loads(zlib.decompress(t[i][j]))
					policy = data[1]
					bestA = np.argmax(policy)
					new_policy = np.zeros_like(policy)
					new_policy[bestA] = 1
					data = (data[0], new_policy, data[2], data[3], data[4], data[5])
					t[i][j] = zlib.compress(pickle.dumps(data), level=1)
			print()

	# breakpoint()

	for t, name in [(training, 'training'), (testing, 'testing')]:
		filename = args.output + '_' + name + '.examples'
		print(f'total size {name} = {sum([len(x) for x in t])} --> writing to {filename}')
		with open(filename, "wb") as f:
			pickle.dump(t, f)
		# print(f'Testing...')
		# with open(filename, "rb") as f:
		#     new_input = pickle.load(f)
		#     print(f'size = {[len(x) for x in new_input]}, total = {sum([len(x) for x in new_input])}')
