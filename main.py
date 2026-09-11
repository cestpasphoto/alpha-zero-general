#!/usr/bin/env python3

import argparse
import logging
import os
import subprocess

import coloredlogs

from Coach import Coach

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.


def run(args):
	from GameSwitcher import import_game
	Game, NNet, players, NUMBER_PLAYERS = import_game(args.game)

	log.debug('Loading %s...', Game.__name__)
	g = Game()

	log.debug('Loading %s...', NNet.__name__)
	nn_args = dict(
		lr=args.learn_rate,
		dropout=args.dropout,
		epochs=args.epochs,
		batch_size=args.batch_size,
		nn_version=args.nn_version,
		learn_rate=args.learn_rate,
		no_compression=args.no_compression,
		q_weight=args.q_weight,
	)
	for kv in args.nn_opt:
		key, sep, raw = kv.partition('=')
		if not sep:
			raise SystemExit(f'[FATAL] --nn-opt expects KEY=VALUE, got "{kv}"')
		key, raw = key.strip(), raw.strip()
		if raw.lower() in ('true', 'false'):
			value = (raw.lower() == 'true')
		else:
			try:
				value = int(raw)
			except ValueError:
				try:
					value = float(raw)
				except ValueError:
					value = raw
		nn_args[key] = value
		#log.info('nn_args[%s] = %r (from --nn-opt)', key, value)
	nnet = NNet(g, nn_args)

	if args.load_model:
		log.info('Loading checkpoint "%s"...', args.load_folder_file)
		nnet.load_checkpoint(os.path.dirname(args.load_folder_file), os.path.basename(args.load_folder_file))

		if os.path.abspath(args.checkpoint) != os.path.abspath(os.path.dirname(args.load_folder_file)):
			import shutil, json
			os.makedirs(args.checkpoint, exist_ok=True)
			
			new_leaderboard_path = os.path.join(args.checkpoint, 'leaderboard.json')
			old_dir = os.path.dirname(args.load_folder_file)
			old_lb_path = os.path.join(old_dir, 'leaderboard.json')

			if not os.path.exists(new_leaderboard_path):
				new_leaderboard = {}
				if os.path.exists(old_lb_path):
					try:
						with open(old_lb_path, 'r') as f:
							old_lb = json.load(f)
						# Nettoyage préventif des clés de l'ancien leaderboard
						valid_old = {m: v for m, v in old_lb.items() if isinstance(v, (int, float)) and not m.startswith('_')}
						top_models = sorted(valid_old, key=valid_old.get, reverse=True)[:3]
						
						for rank, old_name in enumerate(top_models, start=1):
							old_file_path = os.path.join(old_dir, old_name)
							if os.path.exists(old_file_path):
								new_name = f'parent_rank_{rank}.pt'
								shutil.copy(old_file_path, os.path.join(args.checkpoint, new_name))
								new_leaderboard[new_name] = valid_old[old_name]
								log.info(f"Imported {old_name} as {new_name} (Elo: {int(valid_old[old_name])})")
					except Exception as e:
						log.warning(f"Failed to import Elite Vanguard: {e}")

				if not new_leaderboard:
					new_name = 'parent_baseline.pt'
					shutil.copy(args.load_folder_file, os.path.join(args.checkpoint, new_name))
					new_leaderboard[new_name] = 1200.0
					log.info(f"Fallback: League initialized with single {new_name}")

				with open(new_leaderboard_path, 'w') as f:
					json.dump(new_leaderboard, f, indent=2)

		if not args.useray:
			compare_settings(args)
	# else:
	# 	log.warning('Not loading a checkpoint!')

	log.debug('Loading the Coach...')
	c = Coach(g, nnet, args)

	if args.load_model and not args.forget_examples:
		log.info("Loading 'trainExamples' from file...")
		c.loadTrainExamples()

	if not args.useray:
		# Backup code used for this run
		subprocess.run(f'mkdir -p "{args.checkpoint}/"', shell=True)
		subprocess.run(f'cp *py "{args.game}"/*py "{args.checkpoint}/"', shell=True)
		subprocess.run(
			f'[ -f "{args.checkpoint}/settings.txt" ] && mv "{args.checkpoint}/settings.txt" "{args.checkpoint}/settings."`date +%s` ;   echo "{args}" > "{args.checkpoint}/settings.txt"',
			shell=True)

	log.debug('Starting the learning process 🎉')
	c.learn()


# Compare current settings and settings of checkpoints, display main differences
def compare_settings(args):
	settings_file = os.path.join(os.path.dirname(args.load_folder_file), 'settings.txt')

	# Load settings
	if not os.path.isfile(settings_file):
		log.warning('No settings.txt next to the loaded checkpoint: cannot diff the run settings')
		return
	with open(settings_file, 'r') as f:
		previous_args = f.read()

	# Compute differences on dict versions
	previous_args_dict, current_args_dict = vars(eval('argparse.' + previous_args)), vars(args)
	changed_keys = set([k for k in set(list(previous_args_dict.keys()) + list(current_args_dict.keys())) if
	                    previous_args_dict.get(k) != current_args_dict.get(k)])
	for key in ['load_folder_file', 'checkpoint', 'numIters', 'maxlenOfQueue', 'load_model']:
		changed_keys.discard(key)

	if changed_keys:
		log.info('Some option(s) changed compared to loaded checkpoint:')
		for k in changed_keys:
			print(f'{k}: {previous_args_dict.get(k)} --> {current_args_dict.get(k)}')


def profiling(args):
	import cProfile, pstats
	profiler = cProfile.Profile()
	# import yappi
	args.parallel_inferences, args.numIters, args.numEps, args.epochs = 1, 1, 8, 1  # warmup run
	run(args)

	print('\nstart profiling')
	args.parallel_inferences, args.numIters, args.numEps, args.epochs = 1, 1, 8, 1
	# Core of the training
	# yappi.start()
	profiler.enable()
	run(args)
	# yappi.stop()
	profiler.disable()

	# debrief
	profiler.dump_stats('execution.prof')
	print('check dumped stats in execution.prof')
	# Sample code:
	# from pstats import Stats, SortKey
	# p = Stats('execution.prof')
	# p.strip_dirs().sort_stats('cumtime').print_stats(20)
	# p.strip_dirs().sort_stats('tottime').print_stats(10)

	# threads = yappi.get_thread_stats()
	# for thread in threads:
	# 	print("Function stats for (%s) (%d)" % (thread.name, thread.id))  # it is the Thread.__class__.__name__
	# 	yappi.get_func_stats(ctx_id=thread.id).print_all()

	breakpoint()

def main():
	parser = argparse.ArgumentParser(description='tester')
	parser.add_argument('game'                     , action='store', default='splendor', help='The name of the game to simulate')
	parser.add_argument('--checkpoint'      , '-C' , action='store', default='./temp/', help='')
	parser.add_argument('--load-folder-file', '-L' , action='store', default=None     , help='')
	
	parser.add_argument('--numEps'          , '-e' , action='store', default=500  , type=int  , help='Number of complete self-play games to simulate during a new iteration')
	parser.add_argument('--numItersHistory' , '-i' , action='store', default=5   , type=int  , help='')

	parser.add_argument('--numMCTSSims'     , '-m' , action='store', default=1600 , type=int  , help='Number of moves for MCTS to simulate in FULL exploration')
	# --tempThreshold is now the 4th value of --temperature (single source of truth).
	# -T is kept as a DEPRECATED override so existing command lines keep working unchanged.
	parser.add_argument('--tempThreshold'   , '-T' , action='store', default=None , type=int  , help='DEPRECATED: half-life of temp decay, now temperature[3]. If set, overrides it.')
	parser.add_argument('--temperature'     , '-t' , action='store', default=[1.0, 0.1, 1.1, 10.0], type=float, nargs=4, help='[t_begin, t_end, softmax_temp, half_life]. half_life in moves (neg => step). Self-play only')
	parser.add_argument('--cpuct'           , '-c' , action='store', default=1.25 , type=float, help='cpuct value')
	# Replace or add next to --cpuct
	# parser.add_argument('--cpuct'                  , action='store', default=1.25 , type=float, help='cpuct setting')
	# parser.add_argument('--cpuct-init'             , action='store', default=19652, type=int  , help='c_init for log-cpuct')	
	# parser.add_argument('--cpuct-factor'           , action='store', default=1.0. , type=float, help='c_factor for log-cpuct')	
	parser.add_argument('--dirichletAlpha'  , '-d' , action='store', default=-1   , type=float, help='α=0.3 for chess, scaled in inverse proportion to the approximate number of legal moves in a typical position. 0 to disable. -1 for auto.')
	parser.add_argument('--fpu'             , '-f' , action='store', default=0.1  , type=float, help='Value for FPU (first play urgency, using parent-based reduction)')
	parser.add_argument('--fpu-root'               , action='store', default=0.   , type=float, help='Value for FPU at root level (first play urgency, using parent-based reduction)')
	parser.add_argument('--forced-playouts' , '-F' , action='store_true', help='Enabled forced playouts')
	parser.add_argument('--forced-playouts-k', '-k' , action='store', default=1.5  , type=float, help='Multiplier k for forced playouts')
	parser.add_argument('--gumbel'          , '-G' , action='store_true', help='Gumbel root Sequential Halving on self-play full searches. Supersedes Dirichlet noise, forced playouts, PTP and visit-count policy targets at the root. Designed for LOW sim counts (typically -m 32..200); at -m 800 the expected benefit is small. Eval (pit/arena/pnet) is never affected.')
	parser.add_argument('--gumbel-m'               , action='store', default=16   , type=int  , help='Gumbel: max number of root actions considered by Sequential Halving (paper/mctx default: 16)')
	parser.add_argument('--gumbel-cvisit'          , action='store', default=50.0 , type=float, help='Gumbel: c_visit constant of the sigma(Q) transform (paper default: 50)')
	parser.add_argument('--gumbel-cscale'          , action='store', default=1.0  , type=float, help='Gumbel: c_scale constant of the sigma(Q) transform (paper default: 1.0)')
	parser.add_argument('--nn-opt'                 , action='append', default=[], metavar='KEY=VALUE', help='Extra key passed to the net constructor (nn_args). Repeatable. Used for function-preserving growth modules: area_value, attn_pool, graph_mix, graph_layers, extra_layer.')

	parser.add_argument('--learn-rate'      , '-l' , action='store', default=0.0003, type=float, help='')
	parser.add_argument('--epochs'          , '-p' , action='store', default=2    , type=int  , help='')
	parser.add_argument('--batch-size'      , '-b' , action='store', default=32   , type=int  , help='')
	parser.add_argument('--dropout'         , '-D' , action='store', default=0.   , type=float  , help='Dropout value - advised to disable')
	parser.add_argument('--nn-version'      , '-V' , action='store', default=1    , type=int  , help='Which architecture to choose')

	### Advanced params ###
	parser.add_argument('--selfPlayRatio'          , action='store', default=80   , type=int  , help='Percentage of pure self-play games (100 = disable league)')
	parser.add_argument('--leagueSize'             , action='store', default=10   , type=int  , help='Max number of models kept in the league pool')
	parser.add_argument('--q-weight'        , '-q' , action='store', default=0.5  , type=float, help='Weight for mixing Q into value loss')
	parser.add_argument('--arena-gate'      , '-A' , action='store_true', help='Legacy synchronous evaluation mode: after each training iteration, pit the new net against its pre-training snapshot and accept/reject based on --updateThreshold. Disables league sparring. Without this flag (default), checkpoints are saved unconditionally and evaluated asynchronously (pit.py -D), selection is post-hoc.')
	parser.add_argument('--arenaCompare'           , action='store', default=30   , type=int  , help='Arena gate: number of games against the previous net. 30 is a COARSE filter (~±130 Elo): it gates obvious regressions, it does not measure progress')
	parser.add_argument('--arena-sims'             , action='store', default=None , type=int  , help='Arena gate: numMCTSSims used by BOTH sides of the gate (default: --numMCTSSims). Gate cost is ~linear in sims, so lowering it buys games: e.g. 100 games at 400 sims costs about the same wall clock as 30 games at 1200, and resolves ~2x better. Never affects self-play nor pit.py.')
	parser.add_argument('--updateThreshold'        , action='store', default=0.60 , type=float, help='During arena playoff, new neural net will be accepted if threshold or more of games are won')
	parser.add_argument('--ratio-fullMCTS'         , action='store', default=5    , type=int  , help='Ratio of MCTS sims between full and fast exploration')
	parser.add_argument('--prob-fullMCTS'          , action='store', default=0.25 , type=float, help='Probability to choose full MCTS exploration')
	parser.add_argument('--universes'       , '-u' , action='store', default=1    , type=int  , choices=range(9), help='Number of universes (up to 8); will switch between each of them at each rollout. Set to 0 for a deterministic exploration')

	parser.add_argument('--forget-examples'        , action='store_true', help='Do not load previous examples')
	parser.add_argument('--numIters'        , '-n' , action='store', default=50   , type=int, help='')
	parser.add_argument('--stop-after-N-fail', '-s', action='store', default=-2   , type=float, help='Number of consecutive failed arenas that will trigger process stop (-N means N*numItersHistory). Default raised from 5 to 10: under strict parity a 17%% accept rate makes P(5 consecutive rejects)=0.39, so 5 kills healthy runs by luck alone')
	parser.add_argument('--profile'                , action='store_true', help='profiler')
	parser.add_argument('--debug'                  , action='store_true', help='Disable all optimisations to allow easier debugging')
	parser.add_argument('--useray'                 , action='store_true', help='Mode for "ray", disable some messages')
	parser.add_argument('--parallel-inferences','-P',action='store', default=8    , type=int  , help='Size of batch for inferences = nb of threads, set to 1 to disable')
	parser.add_argument('--no-compression'         , action='store_true', help='Prevent using in-memory data compression (huge memory decrease and impact by only by ~1 second per 100k samples), useful for easier debugging')
	parser.add_argument('--no-mem-optim'           , action='store_true', help='Prevent cleaning MCTS tree of old moves during each game')
	
	args = parser.parse_args()
	if args.gumbel:
		if args.forced_playouts:
			# Keep settings.txt truthful: FP would be dead code on full searches anyway
			log.warning('Gumbel enabled: forced playouts / PTP are superseded at the root, disabling --forced-playouts')
			args.forced_playouts = False
		if args.dirichletAlpha != 0:
			log.info('Gumbel enabled: Dirichlet noise will NOT be applied on full searches (superseded by Gumbel sampling); dirichletAlpha is kept only as a record')
	if args.tempThreshold is not None:              # backward-compat: -T overrides temperature[3]
		args.temperature[3] = float(args.tempThreshold)
	args.tempThreshold = int(args.temperature[3])   # canonical value, used as-is across Coach
	if args.arena_gate and args.selfPlayRatio < 100:
		# The two modes answer the same question (which checkpoint to trust) with
		# incompatible machineries; sparring pool comes from the daemon leaderboard
		# which does not exist in gate mode.
		log.warning('Arena gate enabled: league sparring disabled (selfPlayRatio forced to 100)')
		args.selfPlayRatio = 100
	args.maxlenOfQueue = int(2.5e6 / ((
		                                  2 if args.no_compression else 0.5) * args.numItersHistory))  # at most 2GB per process, with each example weighing 2kB (or 0.5kB)
	if args.stop_after_N_fail < 0:
		args.stop_after_N_fail = -args.stop_after_N_fail * args.numItersHistory

	if args.debug:
		args.parallel_inferences = 1
		args.no_compression = True
		args.no_mem_optim = True

	args.load_model = (args.load_folder_file is not None)
	if args.profile:
		profiling(args)
	else:
		if not args.useray:
			print(args)
		run(args)


if __name__ == "__main__":
	main()
