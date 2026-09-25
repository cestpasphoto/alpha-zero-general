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

from Arena import Arena, report_opening_uniqueness
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
		self.pnet = self.nnet.__class__(self.game, self.nnet.args)  # pre-training snapshot, reference of the arena
		self.args = args
		self.mcts = self.new_selfplay_mcts(self.game)
		self.trainExamplesHistory = []  # examples of the args.numItersHistory latest iterations
		self.skipFirstSelfPlay = nnet.requestKnowledgeTransfer
		self.consecutive_failures = 0
		self.nb_threads = self.args.parallel_inferences

	def new_selfplay_mcts(self, game, batch_info=None):
		return MCTS(game, self.nnet, self.args, dirichlet_noise=(self.args.dirichletAlpha!=0),
		            batch_info=batch_info, is_selfplay=True)

	def selfplay_mcts_list(self, game, first_mcts, batch_info=None):
		# Hidden-info games: one tree per seat, a shared tree would pool
		# statistics across seats holding different hidden hands.
		if hasattr(game, 'getObservation'):
			return [first_mcts] + [self.new_selfplay_mcts(game, batch_info) for _ in range(game.num_players - 1)]
		return [first_mcts] * game.num_players

	def executeEpisode(self, my_game=None, mcts_list=None):
		"""
		This function executes one episode of self-play, starting with player 0.
		As the game is played, each full-search turn is added as a training
		example to trainExamples. After the game ends, the outcome of the game
		is used to assign values to each example in trainExamples.

		Returns:
			examples: list of (board, pi, outcome, valids, q), compressed unless
			          --no-compression; outcome is the per-player result of the game
			opening: tuple of the first plies, to measure opening diversity
		"""
		if my_game is None:
			my_game = self.game
		if mcts_list is None:
			mcts_list = self.selfplay_mcts_list(my_game, self.mcts)
		hidden_info = hasattr(my_game, 'getObservation')

		trainExamples = []
		board = my_game.getInitBoard()
		curPlayer = 0
		episodeStep = 0
		opening_sequence = []
		DEPTH_OPENING = 2 * abs(self.args.tempThreshold)   # abs(): tempThreshold < 0 means step mode

		while True:
			episodeStep += 1
			canonicalBoard = my_game.getCanonicalForm(board, curPlayer)
			pi, q, is_full_search = mcts_list[curPlayer].getActionProb(canonicalBoard, temp=1.0)
			action = random_pick(pi, temperature=self.temp_for_selfplay(episodeStep))
			if episodeStep <= DEPTH_OPENING:
				opening_sequence.append(action)

			if is_full_search:
				valids = my_game.getValidMoves(canonicalBoard, 0)
				# Hidden-info games: train on the observation, which is what the net
				# is queried on during the search
				board_for_training = my_game.getObservation(canonicalBoard, 0) if hidden_info else canonicalBoard
				sym = my_game.getSymmetries(board_for_training, pi, valids)
				for b, p, v in sym:
					trainExamples.append([b, p, curPlayer, v, q])

			board, curPlayer = my_game.getNextState(board, curPlayer, action)
			r = my_game.getGameEnded(board, curPlayer)

			if r.any():
				trainExamples = [(
					x[0],                                # board
					x[1],                                # policy
					np.roll(r, -x[2]),                   # winner
					x[3],                                # valids
					x[4],                                # Q estimates
				) for x in trainExamples]

				examples = trainExamples if self.args.no_compression else [zlib.compress(pickle.dumps(x), level=1) for x in trainExamples]
				return examples, tuple(opening_sequence)

	def executeEpisodes_batch(self, i_thread, shared_memory, locks):
		# Execute an episode in a thread until need to evaluate NN
		# then unlock next threads, etc until batch of inferences to do is full
		# then server runs inferences on batch.
		# Each thread loops until receiving a signal to stop
		locks[i_thread].acquire()
		batch_info_nnet = (i_thread, i_thread+self.nb_threads, shared_memory, locks)

		while shared_memory[-1] == 0: # Signal 0 means to continue computing
			my_game = self.game.__class__()
			my_game.getInitBoard()
			first_mcts = self.new_selfplay_mcts(my_game, batch_info_nnet)
			mcts_list = self.selfplay_mcts_list(my_game, first_mcts, batch_info_nnet)
			self.examplesQueue.put(self.executeEpisode(my_game=my_game, mcts_list=mcts_list))

		while shared_memory[-1] == 1: # We received signal 1, wait for other threads to complete
			locks[i_thread+1].release()
			locks[i_thread].acquire()
		locks[i_thread+1].release()

	def executeEpisodes(self):
		iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
		if self.nb_threads == 1:
			nb_episodes = 0
			unique_openings = set()
			for _ in trange(self.args.numEps, desc="Self Play", ncols=120):
				episode_examples, opening = self.executeEpisode()
				iterationTrainExamples += episode_examples
				nb_episodes += 1
				unique_openings.add(opening)
				self.mcts = self.new_selfplay_mcts(self.game)
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
			nb_episodes, max_nb_episodes = 0, self.args.numEps
			unique_openings = set()
			while True:
				sleep(1)
				for _ in range(self.examplesQueue.qsize()):
					episode_examples, opening = self.examplesQueue.get_nowait()
					iterationTrainExamples += episode_examples
					nb_episodes += 1
					unique_openings.add(opening)
					progress.update()
				# Check if we have collected enough samples
				if nb_episodes >= self.args.numEps - self.nb_threads:
					if nb_episodes >= max_nb_episodes:
						shared_memory[-1] = 2 # send signal 2 = all threads can be stopped
						break
					elif shared_memory[-1] == 0:
						max_nb_episodes = nb_episodes + self.nb_threads
						progress.total = max_nb_episodes
						shared_memory[-1] = 1 # send signal 1 = threads can stop after their current episode
			[t.join() for t in threads_list]
			progress.close()

		report_opening_uniqueness(len(unique_openings), nb_episodes, 'Self-play')
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

			# Generate self-play examples for this iteration
			if not self.skipFirstSelfPlay or i > 1:
				iterationTrainExamples = self.executeEpisodes()
				if len(iterationTrainExamples) == self.args.maxlenOfQueue:
					log.warning(f'saturation of elements in iterationTrainExamples...')
				self.trainExamplesHistory.append(iterationTrainExamples)

				if self.args.no_compression:
					nb_valid_moves = [sum(x[3]) for x in iterationTrainExamples]
				else:
					nb_valid_moves = [sum(pickle.loads(zlib.decompress(x))[3]) for x in iterationTrainExamples]
				avg_valid_moves = sum(nb_valid_moves) / len(nb_valid_moves)
				if self.args.dirichletAlpha > 0 and not (1/1.5 < self.args.dirichletAlpha / (10/avg_valid_moves) < 1.5):
					print(f'There are about {avg_valid_moves:.1f} valid moves per state, so I advise to set dirichlet to {10/avg_valid_moves:.1f} instead')

			if self.args.profile:
				return

			if len(self.trainExamplesHistory) > self.args.numItersHistory:
				self.trainExamplesHistory.pop(0)
			
			self.saveTrainExamples()
			trainExamples = []
			for e in self.trainExamplesHistory:
				trainExamples.extend(e)
			shuffle(trainExamples)

			# Snapshot the network BEFORE training: it is the reference of the arena
			self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pt', additional_keys=vars(self.args))
			self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pt')
			self.nnet.train(trainExamples)
			self.arena_gate_step(i)

	def arena_gate_step(self, i):
		"""
		Pit the freshly trained net against its pre-training snapshot (temp.pt,
		already loaded in pnet) and keep it only if it clears args.updateThreshold.

		Both players share the training search profile (including forced playouts
		and policy target pruning when -F is set), always run a full search
		without Dirichlet noise, and sample their moves from the tempered policy.
		At arenaCompare=30 this is a coarse filter (~±130 Elo): it gates obvious
		regressions, it does not measure progress.
		"""
		# Arena takes a factory per side and calls it once per seat for
		# hidden-info games, so each call must build a new MCTS
		def make_gate_player(net):
			def factory():
				mcts = MCTS(self.game, net, self.args)
				def play(x, n):
					probs = mcts.getActionProb(x, temp=self.temp_for_game(n), force_full_search=True)[0]
					return int(np.random.choice(len(probs), p=probs))
				return play
			return factory

		arena = Arena(make_gate_player(self.nnet), make_gate_player(self.pnet), self.game)
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
			# 'best.pt' is the last ACCEPTED checkpoint, not the post-hoc best of the run
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


	# Calculates the exponential decay for temperature during self-plays
	def temp_for_selfplay(self, n):
		t_begin, t_end, half_life = self.args.temperature[0], self.args.temperature[1], self.args.tempThreshold
		if half_life < 0:
			return t_end if n > -half_life else t_begin 
		else:
			return t_end + (t_begin - t_end) * (0.5 ** (n / half_life))

	# Calculates the exponential decay for temperature during test games
	def temp_for_game(self, n):
		t_begin, t_end, half_life = 0.5, 0.0, abs(self.args.tempThreshold)
		return t_end + (t_begin - t_end) * (0.5 ** (n / half_life))

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
