#!/usr/bin/env python3
import importlib
import itertools
import json
import multiprocessing
import os.path
import subprocess
from math import inf

import numpy as np

import Arena
from MCTS import MCTS
from utils import *
from GameSwitcher import import_game

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

game = None
_lock = multiprocessing.Lock()


def create_player(name, args, player_id):
	global game
	global NNet
	global players
	if game is None:
		Game, NNet, players, NUMBER_PLAYERS = import_game(args.game)
		game = Game()
	# all players
	if name == 'random':
		return players.RandomPlayer(game).play, None
	if name == 'greedy':
		return players.GreedyPlayer(game).play, None
	if name == 'human':
		return players.HumanPlayer(game).play, None

	# set default values but will be overloaded when loading checkpoint
	nn_args = dict(lr=None, dropout=0., epochs=None, batch_size=None, nn_version=-1)
	net = NNet(game, nn_args)
	cpt_dir, cpt_file = os.path.split(name)
	additional_keys = net.load_checkpoint(cpt_dir, cpt_file)

	cpuct = additional_keys.get('cpuct')
	cpuct = float(cpuct[0]) if isinstance(cpuct, list) else cpuct
	is_daemon = getattr(args, 'daemon', False)
	strict = getattr(args, 'strict', False)

	if strict:
		# Protocol v1.1 §1: a single EVAL profile, pinned, applied to BOTH players,
		# NEVER inherited from the checkpoint. Any inheritance re-opens the
		# 3200-vs-800 trap and conflates net strength with search hyperparameters.
		if not args.numMCTSSims:
			raise SystemExit('[FATAL] --strict requires an explicit -m/--numMCTSSims (protocol v1.1 §1)')
		mcts_args = dotdict({
			'numMCTSSims'      : args.numMCTSSims,
			'cpuct'            : args.cpuct if args.cpuct else 1.0,
			'fpu'              : args.fpu if getattr(args, 'fpu', None) is not None else 0.1,
			'fpu_root'         : 0.0,
			'universes'        : args.universes if getattr(args, 'universes', None) is not None else additional_keys.get('universes', 1),
			'prob_fullMCTS'    : 1.,      # PCR off in eval
			'forced_playouts'  : False,   # training tool
			'gumbel'           : False,   # training tool
			'forced_playouts_k': 1.5,
			'no_mem_optim'     : False,
		})
		mcts = MCTS(game, net, mcts_args)

		def temp_for_game(n):
			# Explicit eval temperature: 0.5 -> 0, half-life 4 plies (protocol v1.1 §1)
			return 0.5 * (0.5 ** (n / 4.0))

		def player(x, n):
			probs = mcts.getActionProb(x, temp=temp_for_game(n), force_full_search=True)[0]
			return int(np.random.choice(len(probs), p=probs))
		return player, mcts_args

	# Defect 5: detect a silent fallback to the default sim count (the 3200-vs-800 trap)
	sims_from_ckpt = additional_keys.get('numMCTSSims', None)
	sims = args.numMCTSSims if args.numMCTSSims else (sims_from_ckpt if sims_from_ckpt else 100)
	if not args.numMCTSSims and sims_from_ckpt is None:
		print(f"[EVAL WARNING] {name}: numMCTSSims missing from checkpoint, falling back to {sims}. "
		      f"Pass -m explicitly to avoid a silent sim-count mismatch.")

	# Defect 6: --fpu was parsed but never applied; honour it for both players when given
	fpu_cli = args.fpu if getattr(args, 'fpu', None) is not None else None
	fpu_ckpt      = additional_keys.get('fpu')
	fpu_root_ckpt = additional_keys.get('fpu_root', fpu_ckpt)
	mcts_args = dotdict({
		'numMCTSSims'     : sims,
		'fpu'             : fpu_cli if fpu_cli is not None else (0.1 if fpu_ckpt is None else fpu_ckpt),
		'fpu_root'        : 0.0 if is_daemon else (fpu_cli if fpu_cli is not None else (0.0 if fpu_root_ckpt is None else fpu_root_ckpt)),
		'universes'       : additional_keys.get('universes', 1),
		'cpuct'           : args.cpuct if args.cpuct else (1.0 if is_daemon else cpuct),
		'prob_fullMCTS'   : 1.,
		'forced_playouts' : False,
		'gumbel'          : False,  # training-only tool, pinned OFF in eval like FP/Dirichlet (protocol v1.1 §1)
		'forced_playouts_k': additional_keys.get('forced_playouts_k', 1.5),
		'no_mem_optim'    : False,
	})

	mcts = MCTS(game, net, mcts_args)
	def temp_for_game(n):
		if is_daemon:
			return 0.2 if n <= 20 else 0.0
		# Defect 3: half-life read from temperature[3] (merged --tempThreshold), fallback 10
		# for older checkpoints. Was wrongly temperature[2] (softmax temp ~1.1) -> near-greedy
		# play from move ~5, collapsing opening diversity.
		t_begin, t_end = 0.5, 0.0
		half_life = (additional_keys.get('temperature', [])[3:4] or [10])[0]
		return t_end + (t_begin - t_end) * (0.5 ** (n / half_life))
	
	def player(x, n):
		probs = mcts.getActionProb(x, temp=temp_for_game(n), force_full_search=True)[0]
		return int(np.random.choice(len(probs), p=probs))
	return player, mcts_args


def _resolve_player_path(p):
	# Prefer best.pt (post-hoc selected); fall back to latest.pt (most recent checkpoint).
	if os.path.isdir(p):
		for cand in ('best.pt', 'latest.pt'):
			full = os.path.join(p, cand)
			if os.path.exists(full):
				return full
		return os.path.join(p, 'best.pt')  # informative failure downstream
	return p


def _report_decision(result, args, p1_name, p2_name):
	"""
	Decision-grade report (protocol v1.1 §2/§4). Prints the score, its z-score,
	the Elo point estimate and its 95% CI, and the verdict phrased so that a
	null result is reported as 'not detectable at the sprint resolution',
	never as 'no effect'.
	"""
	import math
	oneWon, twoWon, draws = result
	n = oneWon + twoWon + draws
	if n == 0:
		print('[REPORT] no game played')
		return
	s = (oneWon + 0.5 * draws) / n
	# Sample variance of per-game outcomes in {1, 0.5, 0}, draws counted as 1/2
	var = (oneWon * (1 - s) ** 2 + draws * (0.5 - s) ** 2 + twoWon * (0 - s) ** 2) / n
	se = math.sqrt(var / n) if var > 0 else 0.0
	z = (s - 0.5) / se if se > 0 else 0.0

	def to_elo(x):
		x = min(max(x, 1e-6), 1 - 1e-6)
		return 400 * math.log10(x / (1 - x))

	elo = to_elo(s)
	lo, hi = (to_elo(s - 1.96 * se), to_elo(s + 1.96 * se)) if se > 0 else (float('-nan'), float('nan'))

	print()
	print('=' * 72)
	print(f'RESULT  {os.path.basename(os.path.dirname(p1_name))}/{os.path.basename(p1_name)}'
	      f'  vs  {os.path.basename(os.path.dirname(p2_name))}/{os.path.basename(p2_name)}')
	print(f'  {oneWon}-{twoWon} ({draws} draws)   n={n}   score={s:.3f}   z={z:+.2f}')
	print(f'  Elo estimate: {elo:+.0f}   CI95 [{lo:+.0f} ; {hi:+.0f}]')
	if z >= 1.645:
		print(f'  VERDICT: P1 > P2 (one-sided, alpha=5%)')
	elif z <= -1.645:
		print(f'  VERDICT: P2 > P1 (one-sided, alpha=5%)')
	else:
		print(f'  VERDICT: gap NOT DETECTABLE at the sprint resolution (50 Elo).')
		print(f'           This is not "no effect": the true gap lies within the CI above.')
	if n < 400:
		print(f'  [!] n={n} < 400: below the declared decision size (+-34 Elo). '
		      f'Screening only, do not decide on this alone.')
	if game is not None and getattr(game, 'num_players', 2) > 2:
		print(f'  [i] {game.num_players}-player game: P1 holds 1 seat / P2 holds {game.num_players - 1}, '
		      f'alternated by Arena, so parity = score 0.500. The Elo figure uses the usual '
		      f'2-player conversion and is a convention here; the score and z are the primary readout.')
	print('=' * 72)


def play(args):
	players = [_resolve_player_path(p) for p in args.players]

	if not args.useray:
		print(players[0], 'vs', players[1])
	# Defect 5: create_player now also returns the resolved MCTS args (or None for baselines)
	(player1, m1), (player2, m2) = create_player(players[0], args, 0), create_player(players[1], args, 1)
	if m1 is not None and m2 is not None and m1.numMCTSSims != m2.numMCTSSims:
		print(f"[EVAL WARNING] sim-count mismatch: P1={m1.numMCTSSims} vs P2={m2.numMCTSSims}. "
		      f"This pit measures search budget, not network strength. Pin -m for both.")
	if getattr(args, 'strict', False) and not args.useray and m1 is not None and m2 is not None:
		# Protocol v1.1 §1: log the profile of BOTH players at the top of the report
		print(f'EVAL PROFILE p1: {dict(m1)}')
		print(f'EVAL PROFILE p2: {dict(m2)}')
		if dict(m1) != dict(m2):
			raise SystemExit('[FATAL] EVAL profiles differ between players - comparison is not decisional')
		thr = 0.5 + 1.645 * 0.5 / (args.num_games ** 0.5)
		print(f'Decision rule (one-sided, draws=1/2): superiority iff score >= {thr:.3f} '
		      f'({thr * args.num_games:.0f}/{args.num_games}), z >= 1.645')
	human = 'human' in players
	arena = Arena.Arena(player1, player2, game, display=game.printBoard)
	result = arena.playGames(args.num_games, initial_state=args.state, verbose=args.display or human)

	if getattr(args, 'strict', False) and not args.useray:
		_report_decision(result, args, players[0], players[1])

	if args.useray:
		##### Write results in a file
		directory = args.players[1] if os.path.isdir(args.players[1]) else os.path.dirname(args.players[1])
		score = result[1] + result[2] / 2.
		print('Writing score to ' + directory + '/score.txt:  ', score)
		with open(directory + '/score.txt', 'w') as f:
			f.write(f'{score}')
		#####

	return result


def play_age(args):
	players = subprocess.check_output(
		['find', args.compare, '-name', 'best.pt', '-mmin', '-' + str(args.compare_age * 60)])
	players = players.decode('utf-8').strip().split('\n')
	print(players)
	list_tasks = list(itertools.combinations(players, 2))
	plays(list_tasks, args)


def plays(list_tasks, args, callback_results=None):
	import math
	import time
	n = len(list_tasks)
	nb_tasks_per_thread = math.ceil(n / args.max_compare_threads)
	nb_threads = math.ceil(n / nb_tasks_per_thread)
	if nb_threads > 1:
		current_threads_list = subprocess.check_output(['ps', 'ax', '-o', 'command']).decode('utf-8').split('\n')
		idx_thread = sum([1 for t in current_threads_list if 'pit.py' in t]) - 1
		if idx_thread == 0:
			print(f'\t{n} pits to do, splitted in {nb_tasks_per_thread} tasks * {nb_threads} threads')
		if idx_thread < nb_threads - 1:
			print(f'\tPlease call same script {nb_threads - 1 - idx_thread} time(s) more in other console')
		elif idx_thread >= nb_threads:
			print(f'I already have enough processes, exiting current one')
			exit()
	else:
		idx_thread = 0
		if n > 1:
			print(f'\t{n} pits to do')

	last_kbd_interrupt = 0.
	for (p1, p2) in list_tasks[idx_thread::nb_threads]:
		args.players = [p1, p2]
		try:
			game_results = play(args)

		except KeyboardInterrupt:
			now = time.time()
			if now - last_kbd_interrupt < 10:
				exit(0)
			last_kbd_interrupt = now
			print('Skipping this pit (hit CRTL-C once more to stop all)')
		else:
			if callback_results:
				callback_results(p1, p2, game_results, args)


def load_rating(player_file):
	import glicko2
	basename = os.path.splitext(os.path.basename(player_file))[0]
	rating_file = os.path.dirname(player_file) + '/rating' + ('' if basename == 'best' else '_' + basename) + '.json'
	if not os.path.exists(rating_file):
		return glicko2.Player()
	r_dict = json.load(open(rating_file, 'r'))
	return glicko2.Player(rating=r_dict['rating'], rd=r_dict['rd'], vol=r_dict['vol'])


def write_rating(rating_object, player_file):
	basename = os.path.splitext(os.path.basename(player_file))[0]
	rating_file = os.path.dirname(player_file) + '/rating' + ('' if basename == 'best' else '_' + basename) + '.json'
	rating_dict = {'rating': rating_object.rating, 'rd': rating_object.rd, 'vol': rating_object.vol}
	json.dump(rating_dict, open(rating_file, 'w'))


def update_ratings(p1, p2, game_results, args):
	oneWon, twoWon, draws = game_results
	with _lock:
		player1, player2 = load_rating(p1), load_rating(p2)
		p1r, p1rd = player1.rating, player1.rd
		p2r, p2rd = player2.rating, player2.rd
		n = oneWon + twoWon + draws
		player1.update_player([p2r] * n, [p2rd] * n, [1] * oneWon + [0.5] * draws + [0] * twoWon)
		player2.update_player([p1r] * n, [p1rd] * n, [1] * twoWon + [0.5] * draws + [0] * oneWon)
		write_rating(player1, args.players[0])
		write_rating(player2, args.players[1])
		# for p, pname in [(player1, p1), (player2, p2)]:
		# 	print(f'{pname[-20:].rjust(20)} rating={int(p.rating)}±{int(p.rd)}, vol={p.vol:.3e}')

def run_daemon(args):
	import time, glob, json, re, math
	leaderboard_file = os.path.join(args.compare, 'leaderboard.json')
	
	print(f"Starting Asynchronous Evaluator in {args.compare}...")
	while True:
		leaderboard = json.load(open(leaderboard_file)) if os.path.exists(leaderboard_file) else {}
		
		# Récupère les checkpoints qui finissent par un chiffre
		cpts = [f for f in glob.glob(os.path.join(args.compare, 'checkpoint_*.pt')) if re.search(r'checkpoint_\d+\.pt$', f)]
		cpts = sorted(cpts, key=os.path.getmtime)
		new_cpts = [c for c in cpts if os.path.basename(c) not in leaderboard]
		
		if not new_cpts:
			time.sleep(30)
			continue
			
		for skipped_cpt in new_cpts[:-1]:
			leaderboard[os.path.basename(skipped_cpt)] = "skipped"
		
		# On ne garde que le tout dernier checkpoint généré
		cpt = new_cpts[-1] 
		cpt_name = os.path.basename(cpt)
		print(f"\n--- Evaluating {cpt_name} ---")
		cpt_elo = leaderboard.get(cpt_name, 1200.0)
		
		# FILTRE SÉCURISÉ : On isole uniquement les vrais modèles avec un Elo numérique
		valid_models = {m: v for m, v in leaderboard.items() if isinstance(v, (int, float)) and not m.startswith('_')}
		
		# Construit le pool des adversaires avec les modèles valides
		league_pool = [m for m in sorted(valid_models, key=valid_models.get, reverse=True) if valid_models[m] >= 1200.0][:args.league_size]
		candidates = [p for p in league_pool if p != cpt_name]
		
		opponents = list(np.random.choice(candidates, min(args.daemon_opponents, len(candidates)), replace=False)) if candidates else []
		
		# Fallback de sécurité si le pool est vide
		if not opponents and valid_models:
			best_old = max(valid_models, key=valid_models.get)
			if best_old != cpt_name: opponents = [best_old]
		
		# Defect 7: frozen anchors + direct Elo from aggregate score (no K=32 compression).
		# A 30-50 game match is information worth tens of Elo; the old K=32 update moved the
		# rating by only a few points AND mutated the opponents, making the whole board drift.
		elo_estimates = []
		for opp in opponents:
			opp_elo = leaderboard.get(opp, 1200.0)
			print(f"Match: {cpt_name} vs {opp} (anchor Elo: {int(opp_elo)})")

			args.players = [cpt, os.path.join(args.compare, opp)]
			oneWon, twoWon, draws = play(args)

			total_games = oneWon + twoWon + draws
			if total_games == 0: continue
			actual_score = (oneWon + draws * 0.5) / total_games
			# Continuity correction to avoid +/-inf at score 0 or 1
			eps = 0.5 / total_games
			s = min(max(actual_score, eps), 1 - eps)
			# Anchor is frozen: estimate the candidate's Elo, never touch the opponent's
			elo_estimates.append(opp_elo + 400 * math.log10(s / (1 - s)))

		if elo_estimates:
			cpt_elo = sum(elo_estimates) / len(elo_estimates)
		leaderboard[cpt_name] = cpt_elo

		print(f"Current Elo of {cpt_name}: {int(cpt_elo)}")
			
		# --- STAGNATION (EARLY STOPPING) ---
		# Defect 8: compare to the best *candidate* so far (frozen anchors sit high and would
		# trip the counter instantly), with a noise-aware margin (~one CI half-width at ~100
		# games) so the counter tracks real progress, not measurement noise.
		STAGNATION_MARGIN = 50.0
		best_cpt_elo = leaderboard.get("_best_cpt_elo", cpt_elo)
		if cpt_elo >= best_cpt_elo - STAGNATION_MARGIN:
			leaderboard["_stagnation_counter"] = 0
			leaderboard["_best_cpt_elo"] = max(best_cpt_elo, cpt_elo)
		else:
			current_count = leaderboard.get("_stagnation_counter", 0) + 1
			leaderboard["_stagnation_counter"] = current_count
			print(f"Stagnation warning: {current_count}/15 checkpoints without progress "
			      f"(best candidate {int(best_cpt_elo)}, this one {int(cpt_elo)}).")

			if current_count >= 15:
				stop_file = os.path.join(args.compare, 'STOP_TRAINING.flag')
				with open(stop_file, 'w') as f:
					f.write("Stagnation threshold reached.")
				print(">>> STAGNATION LIMIT REACHED. STOP SIGNAL SENT TO COACH. <<<")
				with open(leaderboard_file, 'w') as f: json.dump(leaderboard, f, indent=2)
				exit(0)

		# Sauvegarde l'état unique complet (incluant le compteur de stagnation)
		with open(leaderboard_file, 'w') as f: json.dump(leaderboard, f, indent=2)
		
def play_several_files(args):
	players = args.players[:]  # Copy, because it will be overwritten by plays()
	list_tasks = []
	if args.reference:
		if args.useray:
			list_tasks += list(itertools.product(args.reference, args.players))
		else:
			list_tasks += list(itertools.product(args.players, args.reference))
	if not args.vs_ref_only:
		list_tasks += list(itertools.combinations(args.players, 2))

	if args.ratings:
		plays(list_tasks, args, callback_results=update_ratings)
		for p in players:
			r = load_rating(p)
			name = os.path.basename(os.path.dirname(p)) + (
				'' if os.path.basename(p) == 'best.pt' else (' - ' + os.path.basename(p)))
			print(f'{name[-20:].ljust(20)} rating={int(r.rating)}±{int(r.rd)}, vol={r.vol:.3e}')
	else:
		worst_games = {}

		def show_worst_game(p1, p2, result, _):
			worst_games[p1] = min(result[0], worst_games.get(p1, inf))
			worst_games[p2] = min(result[1], worst_games.get(p2, inf))

		plays(list_tasks, args, callback_results=show_worst_game)
		if len(players) > 3:
			for name, worst_game in worst_games.items():
				print(f'{name:<40}: {worst_game}')


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

	parser.add_argument('--num-games'          , '-n' , action='store', default=30   , type=int  , help='')
	parser.add_argument('--profile'                   , action='store_true', help='enable profiling')
	parser.add_argument('--display'                   , action='store_true', help='display')
	parser.add_argument('--state'              , '-s' , action='store', default="", type=str, help='State to load')

	parser.add_argument('--numMCTSSims'        , '-m' , action='store', default=None, type=int  , help='Number of games moves for MCTS to simulate.')
	parser.add_argument('--cpuct'              , '-c' , action='store', default=None, type=float, help='cpuct value')
	parser.add_argument('--fpu'                , '-f' , action='store', default=None, type=float, help='Value for FPU (first play urgency)')
	parser.add_argument('--strict'             , '-S' , action='store_true', help='Decision-grade pit: pin the EVAL profile of protocol v1.1 §1 on BOTH players (no inheritance from checkpoints), require an explicit -m, PCR/FP/Dirichlet/Gumbel off, explicit eval temperature. Use this for every comparison meant to be decisional.')
	parser.add_argument('--universes'          , '-u' , action='store', default=None, type=int  , help='Override universes for both players (default: value stored in checkpoint)')

	parser.add_argument('game'                        , action='store', default='splendor', help='The name of the game to play')
	parser.add_argument('players'                     , metavar='player', nargs='*', help='list of players to test (either file, or "human" or "random")')
	parser.add_argument('--reference'          , '-r' , metavar='ref'   , nargs='*', help='list of reference players')
	parser.add_argument('--vs-ref-only'        , '-z' , action='store_true', help='Use this option to prevent games between players, only players vs references')
	parser.add_argument('--ratings'            , '-R' , action='store_true', help='Compute ratings based in games results and write ratings on disk')
	parser.add_argument('--useray'                    , action='store_true', help='Mode for "ray", disable some messages')

	parser.add_argument('--compare'            , '-C' , action='store', default='../results', help='Compare all best.pt located in the specified folders')
	parser.add_argument('--compare-age'        , '-A' , action='store', default=None        , help='Maximum age (in hour) of best.pt to be compared', type=int)
	parser.add_argument('--max-compare-threads', '-T' , action='store', default=1           , help='No of threads to run comparison on', type=int)

	parser.add_argument('--daemon'             , '-D' , action='store_true', help='Run as asynchronous evaluator')	
	parser.add_argument('--daemon-opponents'   , '-O' , action='store', default=3, type=int, help='Nb of opponents per evaluation')
	parser.add_argument('--league-size'        , '-L' , action='store', default=20, type=int, help='Max number of models kept in the league pool')
	args = parser.parse_args()
	
	if args.daemon:
		run_daemon(args)
	elif args.profile:
		profiling(args)
	elif args.compare_age:
		play_age(args)
	elif args.reference or len(args.players) > 2:
		play_several_files(args)
	elif len(args.players) == 2:
		play(args)
	else:
		raise Exception('Please specify a player (ai folder, random, greedy or human)')


if __name__ == "__main__":
	main()
