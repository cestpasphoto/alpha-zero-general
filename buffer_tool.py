#!/usr/bin/env python3
"""
buffer_tool.py - inspect a replay buffer (checkpoint.examples) and estimate the
length of the self-play games it comes from.

The buffer does not store game boundaries or move numbers: it is a list of
iterations, each a flat list of examples (board, pi, outcome, valids, q).
Only full-search moves (Playout Cap Randomization) produce examples, each
expanded by getSymmetries(). The length is therefore ESTIMATED per iteration:

    positions = N * mean(1 / c(x))        c(x) = expected copies per position
    plies     = positions / (games * prob_fullMCTS)

c(x) is measured by calling game.getSymmetries() on a sample of buffer
examples (repeated calls when the copy count is random, e.g. Catan keeps
near-forced positions with some probability). "plies" counts every decision
of every player, i.e. the same unit as the temperature half-life -T.

Only the MEAN length is available (no distribution). The estimate is biased
low when self-play stopped early on a full queue (fewer games than numEps):
this is flagged when an iteration holds maxlenOfQueue examples.

Usage:
    python3 buffer_tool.py ../results/RUN/3A_deep               (folder: checkpoint.examples + best.pt)
    python3 buffer_tool.py ../results/RUN/3A_deep/checkpoint.examples --pt ../results/RUN/3A_deep/best.pt
    python3 buffer_tool.py FOLDER --games 1000 --sample 20000
"""

import argparse
import os
import pickle
import sys
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def load_args(pt_path):
	"""(args dict, game name, num_players) stored in a checkpoint, or ({}, None, None)."""
	if pt_path is None or not os.path.isfile(pt_path):
		return {}, None, None
	import torch
	ckpt = torch.load(pt_path, map_location='cpu', weights_only=True)
	return ckpt.get('args') or {}, ckpt.get('game'), ckpt.get('num_players')


def decode(x):
	return pickle.loads(zlib.decompress(x)) if isinstance(x, (bytes, bytearray)) else x


def copies_per_position(game, ex, repeats, probe=8):
	"""(expected number of examples getSymmetries() makes for this example's position, random?).
	A few probe calls detect a random count (e.g. Catan keeps near-forced positions with some
	probability); only then is the mean taken over many calls, so that 1/mean stays nearly unbiased."""
	board, pi, _, valids, _ = ex
	counts = [len(game.getSymmetries(board, pi, valids)) for _ in range(probe)]
	if len(set(counts)) == 1 and counts[0] > 0:
		return float(counts[0]), False
	counts += [len(game.getSymmetries(board, pi, valids)) for _ in range(repeats - probe)]
	return max(float(np.mean(counts)), 1. / repeats), True   # a kept example had at least one copy


def main():
	parser = argparse.ArgumentParser(description='Replay buffer inspection: estimated self-play game length per iteration')
	parser.add_argument('path', help='checkpoint.examples, or the folder holding it')
	parser.add_argument('--pt', default=None, help='checkpoint giving the run args (default: best.pt, else temp.pt, next to the buffer)')
	parser.add_argument('--game', default=None, help='game package (default: read from the checkpoint)')
	parser.add_argument('--games', '-e', type=int, default=None, help='games per iteration (default: numEps of the checkpoint)')
	parser.add_argument('--prob-full', type=float, default=None, help='prob_fullMCTS (default: from the checkpoint, else 0.25)')
	parser.add_argument('--sample', type=int, default=5000, help='examples sampled per iteration to measure symmetry copies (default 5000)')
	parser.add_argument('--repeats', type=int, default=400, help='getSymmetries() calls per sampled example when its copy count is random (default 400)')
	parser.add_argument('--seed', type=int, default=0)
	args = parser.parse_args()

	buf_path = os.path.join(args.path, 'checkpoint.examples') if os.path.isdir(args.path) else args.path
	folder = os.path.dirname(os.path.abspath(buf_path))
	pt_path = args.pt
	if pt_path is None:
		pt_path = next((os.path.join(folder, f) for f in ('best.pt', 'temp.pt') if os.path.isfile(os.path.join(folder, f))), None)

	run_args, game_name, ckpt_players = load_args(pt_path)
	game_name = args.game or game_name
	if game_name is None:
		sys.exit('[buffer_tool] game unknown: pass --game (or --pt with a checkpoint that stores it)')
	n_games = args.games or run_args.get('numEps')
	if n_games is None:
		sys.exit('[buffer_tool] number of games per iteration unknown: pass --games')
	p_full = args.prob_full if args.prob_full is not None else run_args.get('prob_fullMCTS', 0.25)
	maxlen = run_args.get('maxlenOfQueue')
	temperature = run_args.get('temperature')
	half_life = temperature[3] if isinstance(temperature, (list, tuple)) and len(temperature) > 3 else run_args.get('tempThreshold')

	from GameSwitcher import import_game
	Game, _, _, code_players = import_game(game_name)
	game = Game()
	if ckpt_players is not None and ckpt_players != code_players:
		sys.exit(f'[buffer_tool] STOP: checkpoint has {ckpt_players} players, code is set to NUMBER_PLAYERS={code_players}')

	with open(buf_path, 'rb') as f:
		history = pickle.load(f)

	print(f'buffer   {buf_path}')
	print(f'args     {pt_path or "-"}')
	print(f'game     {game_name}, {code_players} players, games/iteration {n_games}, prob_fullMCTS {p_full}, current -T {half_life}')
	print()
	print(f'{"iter":>4} {"examples":>9} {"copies":>7} {"positions":>9} {"plies/game":>10} {"per player":>10} {"legal":>6}  note')

	rng = np.random.default_rng(args.seed)
	random_copies = False
	all_plies, all_weights = [], []
	for it, examples in enumerate(history):
		n = len(examples)
		if n == 0:
			print(f'{it:>4} {0:>9}  (empty)')
			continue
		idx = rng.choice(n, size=min(args.sample, n), replace=False)
		sampled = [decode(examples[i]) for i in idx]
		measured = [copies_per_position(game, ex, args.repeats) for ex in sampled]
		c = np.array([m[0] for m in measured])
		random_copies = random_copies or any(m[1] for m in measured)
		positions = n * float(np.mean(1. / c))
		plies = positions / (n_games * p_full)
		legal = float(np.mean([np.sum(ex[3]) for ex in sampled]))
		note = 'full queue: fewer games than numEps, length underestimated' if maxlen and n >= maxlen else ''
		print(f'{it:>4} {n:>9} {n / positions:>7.2f} {positions:>9.0f} {plies:>10.1f} {plies / code_players:>10.1f} {legal:>6.1f}  {note}')
		all_plies.append(plies)
		all_weights.append(n_games)

	if all_plies:
		mean_plies = float(np.average(all_plies, weights=all_weights))
		lo, hi = max(5, round(0.10 * mean_plies)), max(5, round(0.15 * mean_plies))
		print()
		print(f'mean     {mean_plies:.1f} plies per game (all players), {mean_plies / code_players:.1f} per player')
		print(f'rule     -T = 10-15 % of the length, min 5  ->  -T {lo} to {hi}  (current {half_life})')
		if random_copies:
			print('note     random copy count per position (getSymmetries): copies measured over repeated calls')


if __name__ == '__main__':
	main()
