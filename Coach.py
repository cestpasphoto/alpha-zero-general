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
import json

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

	def executeEpisode(self, my_mcts=None, my_game=None, mcts_list=None, players_to_save=None):
		"""
		This function executes one episode of self-play, starting with player 1.
		As the game is played, each turn is added as a training example to
		trainExamples. The game is played till the game ends. After the game
		ends, the outcome of the game is used to assign values to each example
		in trainExamples.

		Returns:
			trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
						   pi is the MCTS informed policy vector, v is +1 if
						   the player eventually won the game, else -1.
		"""
		if isinstance(my_mcts, list):
			mcts_list = my_mcts
			my_mcts = None

		if my_game is None: my_game = self.game
		if mcts_list is None: 
			if my_mcts is None: my_mcts = getattr(self, 'mcts', None)
			mcts_list = [my_mcts] * my_game.num_players
		if players_to_save is None: 
			players_to_save = list(range(my_game.num_players))

		trainExamples = []
		board = my_game.getInitBoard()
		curPlayer = 0
		episodeStep = 0
		episode_metrics = {"max_depth": [], "avg_new_depth": [], "new_nodes": [], "entropy": [], "confidence": [], "root_coverage": []}
		opening_sequence = []
		DEPTH_OPENING = 2 * self.args.tempThreshold

		while True:
			episodeStep += 1
			canonicalBoard = my_game.getCanonicalForm(board, curPlayer)
			
			my_mcts = mcts_list[curPlayer]
			is_saving = (curPlayer in players_to_save)
			
			# pnet joue sans exploration (temp=0.0) et en full_search
			temp = 1.0 if is_saving else 0.0
			force_full = not is_saving
			
			pi, q, is_full_search, metrics = my_mcts.getActionProb(canonicalBoard, temp=temp, force_full_search=force_full)
			action = random_pick(pi, temperature=self.temp_for_selfplay(episodeStep) if is_saving else 0.0)
			
			if episodeStep <= DEPTH_OPENING:
				opening_sequence.append(action)

			if is_full_search and is_saving:
				for k, v in metrics.items():
					episode_metrics[k].append(v)
				valids = my_game.getValidMoves(canonicalBoard, 0)
				sym = my_game.getSymmetries(canonicalBoard, pi, valids)
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
				avg_metrics = {k: np.mean(v) for k, v in episode_metrics.items()} if episode_metrics["max_depth"] else {k: 0.0 for k in episode_metrics.keys()}
				avg_metrics["opening"] = tuple(opening_sequence)
				return examples, avg_metrics

	def executeEpisodes_batch(self, i_thread, shared_memory, locks):
		# Execute an episode in a thread until need to evaluate NN
		# then unlock next threads, etc until batch of inferences to do is full
		# then server runs inferences on batch.
		# Each thread loops until receiving a signal to stop
		locks[i_thread].acquire()
		batch_info_nnet = (i_thread, i_thread+self.nb_threads, shared_memory, locks, 0)
		batch_info_pnet = (i_thread, i_thread+self.nb_threads, shared_memory, locks, 1)

		while shared_memory[-1] == 0: # Signal 0 means to continue computing
			my_game = self.game.__class__()
			my_game.getInitBoard()
			
			# Tire les dés pour ce match précis
			is_asymmetric = (np.random.rand() >= (self.args.selfPlayRatio / 100.0)) and getattr(self, 'pnet_loaded', False) and my_game.num_players > 1
			players_to_save = list(range(my_game.num_players))
			
			if is_asymmetric:
				pnet_player = np.random.randint(my_game.num_players)
				players_to_save.remove(pnet_player)
				from copy import deepcopy
				pnet_args = deepcopy(self.args)
				pnet_args.forced_playouts = False # Désactive l'exploration forcée pour l'évaluateur
				
				mcts_list = []
				for p in range(my_game.num_players):
					if p == pnet_player:
						mcts_list.append(MCTS(my_game, self.pnet, pnet_args, dirichlet_noise=False, batch_info=batch_info_pnet))
					else:
						mcts_list.append(MCTS(my_game, self.nnet, self.args, dirichlet_noise=(self.args.dirichletAlpha!=0), batch_info=batch_info_nnet))
			else:
				mcts_nnet = MCTS(my_game, self.nnet, self.args, dirichlet_noise=(self.args.dirichletAlpha!=0), batch_info=batch_info_nnet)
				mcts_list = [mcts_nnet] * my_game.num_players

			episode_examples, episode_metrics = self.executeEpisode(my_game=my_game, mcts_list=mcts_list, players_to_save=players_to_save)
			self.examplesQueue.put((episode_examples, episode_metrics))

		while shared_memory[-1] == 1: # We received signal 1, wait for other threads to complete
			locks[i_thread+1].release()
			locks[i_thread].acquire()
		locks[i_thread+1].release()

	def executeEpisodes(self):
		iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
		if self.nb_threads == 1:
			total_metrics = {"max_depth": 0, "avg_new_depth": 0, "new_nodes": 0, "entropy": 0, "confidence": 0, "root_coverage": 0}
			completed_episodes = 0
			unique_openings = set()
			t = trange(self.args.numEps, desc="Self Play", ncols=120)
			for _ in t:
				episode_examples, episode_metrics = self.executeEpisode()
				iterationTrainExamples += episode_examples
				completed_episodes += 1
				# log.info({k:v/completed_episodes for k,v in total_metrics.items()})
				for k in total_metrics:
					total_metrics[k] += episode_metrics[k]
				unique_openings.add(episode_metrics["opening"])
				t.set_postfix(
					d_max=f"{total_metrics['max_depth']/completed_episodes:.1f}",
					d_avg=f"{total_metrics['avg_new_depth']/completed_episodes:.1f}",
					# n=f"{total_metrics['new_nodes']/completed_episodes:.0f}",
					ent=f"{total_metrics['entropy']/completed_episodes:.2f}",
					conf=f"{total_metrics['confidence']/completed_episodes:.2f}",
					cov=f"{total_metrics['root_coverage']/completed_episodes:.0%}",
					uniq=f"{len(unique_openings)/completed_episodes:.0%}",
					refresh=False
				)
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
			pnet_to_pass = self.pnet if getattr(self, 'pnet_loaded', False) else None
			threads_list.append(Thread(target=self.nnet.predict_server, args=(self.nb_threads, shared_memory, locks, pnet_to_pass)))
			[t.start() for t in threads_list]

			progress = tqdm(total=self.args.numEps, desc="Self Play", ncols=120, smoothing=0.1, disable=None)
			nb_examples, max_nb_episodes = 0, self.args.numEps
			total_metrics = {"max_depth": 0, "avg_new_depth": 0, "new_nodes": 0, "entropy": 0, "confidence": 0, "root_coverage": 0}
			unique_openings = set()
			while True:
				sleep(1)
				for _ in range(self.examplesQueue.qsize()):
					episode_examples, episode_metrics = self.examplesQueue.get_nowait()
					iterationTrainExamples += episode_examples
					nb_examples += 1
					for k in total_metrics:
						total_metrics[k] += episode_metrics[k]
					unique_openings.add(episode_metrics["opening"])
					progress.set_postfix(
						d_max=f"{total_metrics['max_depth']/nb_examples:.1f}",
						d_avg=f"{total_metrics['avg_new_depth']/nb_examples:.1f}",
						# n=f"{total_metrics['new_nodes']/nb_examples:.0f}",
						ent=f"{total_metrics['entropy']/nb_examples:.2f}",
						conf=f"{total_metrics['confidence']/nb_examples:.2f}",
						cov=f"{total_metrics['root_coverage']/nb_examples:.0%}",
						uniq=f"{len(unique_openings)/nb_examples:.0%}",
						refresh=False
					)
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
			stop_file = os.path.join(self.args.checkpoint, 'STOP_TRAINING.flag')
			if os.path.exists(stop_file):
				log.warning("Stop signal received from Evaluator (Stagnation). Halting training gracefully.")
				break

			# 1. Sélectionne le sparring partner de l'itération si la ligue est activée
			self.pnet_loaded = False
			if self.args.selfPlayRatio < 100:
				leaderboard_file = os.path.join(self.args.checkpoint, 'leaderboard.json')
				if os.path.exists(leaderboard_file):
					try:
						import json
						with open(leaderboard_file, 'r') as f:
							leaderboard = json.load(f)
						if leaderboard:
							# Trie par Elo décroissant et garde les 20 meilleurs
							top_models = sorted(leaderboard, key=leaderboard.get, reverse=True)[:self.args.leagueSize]
							selected = np.random.choice(top_models)
							self.pnet.load_checkpoint(folder=self.args.checkpoint, filename=selected)
							self.pnet_loaded = True
							log.info(f"League Active: Loaded {selected} (Elo: {int(leaderboard[selected])}) as sparring partner.")
					except Exception as e:
						log.warning(f"Could not load {leaderboard_file}: {e}")

			# 2. Génération des exemples (Mélange 80% self-play / 20% ligue géré en interne)
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

			# 3. Entraînement du modèle courant
			self.nnet.train(trainExamples)

			# 5. Sauvegarde finale Asynchrone (Plus de matchs Arena synchrones !)
			log.info(f'Iter #{i} - Training completed. Saving Checkpoint.')
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
