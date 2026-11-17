#!/usr/bin/env python3
"""Train a game from scratch with a fixed, game-independent staged recipe (v3).

Stages, in <checkpoint>/<stage>/ (one main.py run per block of iterations):

  1_takeoff  m100, -q --q-takeoff, 100 games x 2 iterations, then up to --takeoff-extra single iterations
             of 200 games until the arenas show a clear takeoff: one score >= --takeoff-score, or two
             scores >= --takeoff-score2, on >= --takeoff-decisive decisive games. Without takeoff, the stage restarts
             from scratch (--takeoff-restarts times), then goes on with a warning.
  2_m100     m100, --games games per iteration, -q --q-m100, blocks of --block iterations.
             After each block, pit (--pit-games games, m100) of the block end against the
             previous block end (or the takeoff net): go on while the score is
             >= --m100-threshold, at most --m100-blocks blocks.
  3_m800     m800, --games games per iteration, -q --q-m800, -F (unless --no-F), same blocks:
             go on while the block-end pit (m800) is >= --m800-threshold, at most
             --m800-blocks blocks. Raising --m800-blocks with --resume extends a run that
             stopped on the block limit.

The net kept at the end of 2_m100 / 3_m800 is the last block end, unless its pit against
the previous block end is below 0.5 (then the previous block end). It is copied to final.pt.
One training config (--train, --train-m800 for 3_m800). Inside a block, the arena gate of Coach
stays the ratchet (-s is disabled, the block-end pits decide when to stop).

Every decision is stored in <stage>/stage.json, so --resume continues where it stopped.
Summary in <checkpoint>/scenario.tsv. Arena results come from the arenas.csv that Coach
appends in each block folder.

Example:
    ./scenario.py catan -V 13 -C ../results/scn_catan2 --no-F -- -u 2 -P 8
"""
import argparse
import glob
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_TRAIN = '-b 32 -l 3e-3 -p 2'
STAGES = ['takeoff', 'm100', 'm800']
STAGE_DIRS = {'takeoff': '1_takeoff', 'm100': '2_m100', 'm800': '3_m800'}
FORBIDDEN_COMMON = {'-C', '--checkpoint', '-L', '--load-folder-file', '-V', '--nn-version', '-b', '--batch-size',
                    '-l', '--learn-rate', '-p', '--epochs', '-m', '--numMCTSSims', '--forget-examples',
                    '-e', '--numEps', '-s', '--stop-after-N-fail', '-n', '--numIters', '-q', '--q-weight', '-F',
                    '--forced-playouts'}
NO_STOP = '1000'   # -s value that never stops a block early
RE_PIT = re.compile(r'^RESULT\s+(\S+)\s+vs\s+(\S+)')
RE_SCORE = re.compile(r'score=([\d.]+)')


# ----------------------------------------------------------------------------- helpers

def say(msg):
	print(f'\n[scenario] {msg}', flush=True)


def option_value(tokens, *names):
	for i, t in enumerate(tokens[:-1]):
		if t in names:
			return tokens[i + 1]
	return None


def elo(score):
	score = min(max(score, 1e-3), 1 - 1e-3)
	return 400 * math.log10(score / (1 - score))


def newest_net(folder):
	"""Current net of a folder: best.pt if an arena was ever accepted there (a rejection reloads temp.pt, which then
	holds the same weights), otherwise temp.pt (the starting net)."""
	best, temp = os.path.join(folder, 'best.pt'), os.path.join(folder, 'temp.pt')
	if os.path.isfile(best) and any(a[3] for a in read_arenas(folder)):
		return best
	return temp if os.path.isfile(temp) else (best if os.path.isfile(best) else None)


def load_state(folder):
	path = os.path.join(folder, 'stage.json')
	if os.path.isfile(path):
		with open(path) as f:
			return json.load(f)
	return {}


def save_state(folder, state):
	os.makedirs(folder, exist_ok=True)
	path = os.path.join(folder, 'stage.json')
	with open(path + '.tmp', 'w') as f:
		json.dump(state, f, indent=1)
	os.replace(path + '.tmp', path)


def read_arenas(folder):
	"""[[new_wins, old_wins, draws, accepted], ...] from the arenas.csv written by Coach."""
	path = os.path.join(folder, 'arenas.csv')
	if not os.path.isfile(path):
		return []
	with open(path) as f:
		rows = [line.strip().split(',') for line in f if line.strip()]
	return [[int(r[1]), int(r[2]), int(r[3]), r[4] == '1'] for r in rows]


def arena_summary(arenas):
	"""'accepted/total' of a list of arenas."""
	return f'{sum(a[3] for a in arenas)}/{len(arenas)}'


def stamp_part(folder, state):
	"""Rename the numbered checkpoints of the previous main.py run: a new run restarts at checkpoint_1."""
	part = state.get('parts', 0) + 1
	for f in glob.glob(os.path.join(folder, 'checkpoint_*.pt')):
		os.rename(f, os.path.join(folder, f'part{part}_' + os.path.basename(f)))
	state['parts'] = part


# ----------------------------------------------------------------------------- external tools

def run_main(args, folder, tokens, train, load, forget_examples, common):
	cmd = [sys.executable, os.path.join(HERE, 'main.py'), args.game, '-V', str(args.nn_version), '-C', folder]
	if load:
		cmd += ['-L', load]
	if forget_examples:
		cmd += ['--forget-examples']
	cmd += tokens + shlex.split(train) + common
	say('$ ' + shlex.join(cmd))
	t0 = time.time()
	ret = subprocess.run(cmd, cwd=HERE).returncode   # stdout/stderr inherited: live console
	if ret != 0:
		raise SystemExit(f'[scenario] main.py exited with code {ret}, stopping (rerun with --resume)')
	return (time.time() - t0) / 60


def pit_score(args, player, reference, m, n_games, common):
	"""pit.py --strict of player against reference: score of the player."""
	cmd = [sys.executable, os.path.join(HERE, 'pit.py'), args.game, player, '-r', reference, '-z', '--strict',
	       '-m', str(m), '-n', str(n_games)]
	u = option_value(common, '-u', '--universes')
	if u is not None:
		cmd += ['-u', u]
	say('$ ' + shlex.join(cmd))
	proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, text=True, bufsize=1,
	                        env=dict(os.environ, PYTHONUNBUFFERED='1'))   # stderr inherited: live progress
	score, current = None, False
	for line in proc.stdout:
		print(line, end='', flush=True)
		if RE_PIT.search(line.strip()):
			current = True
		elif current and RE_SCORE.search(line):
			score, current = float(RE_SCORE.search(line).group(1)), False
	if proc.wait() != 0 or score is None:
		raise SystemExit(f'[scenario] pit.py failed (code {proc.returncode}), rerun with --resume')
	return score


# ----------------------------------------------------------------------------- stages

def stage_dir(args, stage):
	folder = os.path.join(args.checkpoint, STAGE_DIRS[stage])
	if os.path.isdir(folder) and os.listdir(folder) and not args.resume:
		raise SystemExit(f'[scenario] {folder} already exists: use --resume or another -C')
	return folder


def takeoff_iteration(args, arenas):
	"""1-based index of the arena that confirms the takeoff, None if there is none yet: one arena with a share of
	decisive games >= --takeoff-score, or a second one >= --takeoff-score2 (each with >= --takeoff-decisive decisive games)."""
	n_second = 0
	for i, (new, old, _, _) in enumerate(arenas, start=1):
		if new + old < args.takeoff_decisive:
			continue
		share = new / (new + old)
		if share >= args.takeoff_score:
			return i
		if share >= args.takeoff_score2:
			n_second += 1
			if n_second >= 2:
				return i
	return None


def run_takeoff(args, start, forget, rows, common):
	"""Short m100 runs until one arena shows that training took off; restart from scratch if it never does."""
	parts = [(100, 2)] + [(200, 1)] * args.takeoff_extra
	first_attempt = len(glob.glob(os.path.join(args.checkpoint, f'{STAGE_DIRS["takeoff"]}_failed*')))
	for attempt in range(first_attempt, args.takeoff_restarts + 1):
		folder = stage_dir(args, 'takeoff')
		state = load_state(folder)
		if state.get('done'):
			break
		say(f'===== takeoff (attempt {attempt + 1}): up to {" + ".join(f"{n}x{e}" for e, n in parts)} games =====')
		if not state:
			state = dict(parts_done=0, minutes=0., start=start, forget=forget)
		while state['parts_done'] < len(parts):
			e, n = parts[state['parts_done']]
			if state['parts_done'] == 0 and not state.get('launched'):
				load, fgt = state['start'], state['forget']
			else:
				stamp_part(folder, state)
				load, fgt = newest_net(folder), False
			state['launched'] = True
			save_state(folder, state)
			tokens = ['-m', '100', '-e', str(e), '-q', str(args.q_takeoff), '-n', str(n), '-s', NO_STOP]
			state['minutes'] += run_main(args, folder, tokens, args.train, load, fgt, common)
			state['parts_done'] += 1
			state['takeoff_iter'] = takeoff_iteration(args, read_arenas(folder))
			save_state(folder, state)
			if state['takeoff_iter']:
				break
		if state.get('takeoff_iter'):
			say(f'takeoff at iteration {state["takeoff_iter"]}')
			break
		if attempt < args.takeoff_restarts:
			failed = os.path.join(args.checkpoint, f'{STAGE_DIRS["takeoff"]}_failed{attempt + 1}')
			say(f'no takeoff after {sum(n for _, n in parts)} iterations: kept in {failed}, restarting from scratch')
			os.rename(folder, failed)
		else:
			state['warning'] = 'NO TAKEOFF'
			say('!' * 72 + f'\n[scenario] WARNING: no takeoff after {args.takeoff_restarts + 1} attempts.'
			    ' Going on anyway: check the game logic / network, or lower --takeoff-score.\n' + '!' * 72)
	state['done'] = True
	state['final'] = newest_net(folder)
	save_state(folder, state)
	rows.append(dict(stage='takeoff', block='-', games='100/200', m=100, q=args.q_takeoff,
	                 arenas=arena_summary(read_arenas(folder)), minutes=state['minutes'], pit_minutes=0, score=None,
	                 note=state.get('warning') or f'takeoff it.{state.get("takeoff_iter")}'))
	write_tsv(args, rows)
	return state['final']


def run_block(args, folder, m, q, load, common):
	"""One block of --block iterations in its own folder, resumed if it was interrupted."""
	state = load_state(folder)
	tokens = ['-m', str(m), '-e', str(args.games), '-q', str(q), '-s', NO_STOP] + (['-F'] if m == 800 and not args.no_F else [])
	done = len(read_arenas(folder))
	while done < args.block:
		if state.get('launched'):
			stamp_part(folder, state)
			load = newest_net(folder) or load
			say(f'resuming interrupted block from {load} ({done}/{args.block} iterations done)')
		state['launched'] = True
		save_state(folder, state)
		train = args.train_m800 if m == 800 else args.train
		state['minutes'] = state.get('minutes', 0) + run_main(args, folder, tokens + ['-n', str(args.block - done)],
		                                                       train, load, False, common)
		save_state(folder, state)
		done = len(read_arenas(folder))
	return state.get('minutes', 0)


def run_blocks(args, stage, start, rows, common):
	"""Blocks of iterations until the block-end pit against the previous block end falls below the threshold."""
	m, q = (100, args.q_m100) if stage == 'm100' else (800, args.q_m800)
	threshold, max_blocks = (args.m100_threshold, args.m100_blocks) if stage == 'm100' else (args.m800_threshold, args.m800_blocks)
	folder = stage_dir(args, stage)
	state = load_state(folder) or dict(start=start, blocks=[])
	save_state(folder, state)
	say(f'===== {STAGE_DIRS[stage]}: m{m}, {args.games} games, -q {q}, blocks of {args.block} it., '
	    f'go on while pit >= {threshold}, at most {max_blocks} blocks =====')
	if state.get('done') and state.get('reason') == 'block limit' and len(state['blocks']) < max_blocks:
		say(f'extending: {len(state["blocks"])} blocks done, limit raised to {max_blocks}')
		state['done'] = False
	while not state.get('done'):
		k = len(state['blocks'])
		previous = state['blocks'][-1]['net'] if state['blocks'] else state['start']
		bfolder = os.path.join(folder, f'b{k + 1}')
		say(f'{STAGE_DIRS[stage]} block {k + 1}: {args.block} iterations from {previous}')
		minutes = run_block(args, bfolder, m, q, previous, common)
		net = newest_net(bfolder)
		say(f'{STAGE_DIRS[stage]} block {k + 1}: pit of its end against {previous}')
		t0 = time.time()
		score = pit_score(args, net, previous, m, args.pit_games, common)
		block = dict(folder=bfolder, net=net, previous=previous, minutes=minutes, score=score,
		             pit_minutes=(time.time() - t0) / 60, arenas=arena_summary(read_arenas(bfolder)))
		state['blocks'].append(block)
		say(f'{STAGE_DIRS[stage]} block {k + 1}: score {score:.3f} ({elo(score):+.0f} Elo) vs previous block end, '
		    f'threshold {threshold}')
		if score < threshold:
			state['done'], state['reason'] = True, f'pit {score:.3f} < {threshold}'
		elif len(state['blocks']) >= max_blocks:
			state['done'], state['reason'] = True, 'block limit'
		save_state(folder, state)
	last = state['blocks'][-1]
	state['final'] = last['net'] if last['score'] >= 0.5 else last['previous']
	save_state(folder, state)
	say(f'{STAGE_DIRS[stage]} done ({state["reason"]}): keeping {state["final"]}')
	for i, b in enumerate(state['blocks'], start=1):
		rows.append(dict(stage=stage, block=str(i), games=str(args.games), m=m, q=q, arenas=b['arenas'],
		                 minutes=b['minutes'], pit_minutes=b.get('pit_minutes', 0), score=b['score'], note=state['reason'] if i == len(state['blocks']) else ''))
	write_tsv(args, rows)
	return state['final']


# ----------------------------------------------------------------------------- report

def write_tsv(args, rows):
	with open(os.path.join(args.checkpoint, 'scenario.tsv'), 'w') as f:
		f.write('\t'.join(['stage', 'block', 'games', 'm', 'q', 'accepted', 'minutes', 'pit_minutes', 'pit_vs_previous', 'elo', 'note']) + '\n')
		for r in rows:
			sc = '' if r['score'] is None else f'{r["score"]:.3f}'
			el = '' if r['score'] is None else f'{elo(r["score"]):+.0f}'
			f.write('\t'.join([r['stage'], r['block'], r['games'], str(r['m']), str(r['q']), r['arenas'],
			                   f'{r["minutes"]:.0f}', f'{r["pit_minutes"]:.0f}', sc, el, r['note']]) + '\n')


def print_summary(args, rows, final):
	say(f'summary (training config {args.train}, m800: {args.train_m800})')
	print(f'{"stage":8} {"block":>5} {"games":>7} {"m":>4} {"q":>4} {"acc":>6} {"min":>6} {"pitmin":>6} {"pit":>6} {"elo":>6}  note')
	for r in rows:
		sc = '-' if r['score'] is None else f'{r["score"]:.3f}'
		el = '-' if r['score'] is None else f'{elo(r["score"]):+.0f}'
		print(f'{r["stage"]:8} {r["block"]:>5} {r["games"]:>7} {r["m"]:>4} {r["q"]:>4} {r["arenas"]:>6} '
		      f'{r["minutes"]:>6.0f} {r["pit_minutes"]:>6.0f} {sc:>6} {el:>6}  {r["note"]}')
	train_h, pit_h = sum(r['minutes'] for r in rows) / 60, sum(r['pit_minutes'] for r in rows) / 60
	print(f'TOTAL TIME: {train_h + pit_h:.1f} h  (training {train_h:.1f} h + block-end pits {pit_h:.1f} h)')
	print(f'final network: {final} (copied to {os.path.join(args.checkpoint, "final.pt")})', flush=True)


# ----------------------------------------------------------------------------- main

def main():
	argv = sys.argv[1:]
	common = []
	if '--' in argv:
		common, argv = argv[argv.index('--') + 1:], argv[:argv.index('--')]
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument('game')
	parser.add_argument('--nn-version', '-V', required=True, type=int, help='network version (mandatory)')
	parser.add_argument('--checkpoint', '-C', required=True, help='scenario folder, one sub-folder per stage')
	parser.add_argument('--load', '-L', default=None,
	                    help='warm-start checkpoint for the first stage (its checkpoint.examples is used if present)')
	parser.add_argument('--start-at', choices=STAGES, default='takeoff', help='first stage to run (m100/m800 need --load)')
	parser.add_argument('--until', choices=STAGES, default='m800', help='last stage to run')
	parser.add_argument('--train', default=DEFAULT_TRAIN, help=f'training config of 1_takeoff and 2_m100 (default: {DEFAULT_TRAIN})')
	parser.add_argument('--train-m800', default=None, help='training config of 3_m800 (default: same as --train)')
	parser.add_argument('--games', type=int, default=1000, help='games per iteration after the takeoff')
	parser.add_argument('--q-takeoff', type=float, default=0., help='-q of 1_takeoff (value target = z only while the net is random)')
	parser.add_argument('--q-m100', type=float, default=1.0, help='-q of 2_m100')
	parser.add_argument('--q-m800', type=float, default=1.0, help='-q of 3_m800')
	parser.add_argument('--no-F', action='store_true', help='no forced playouts in 3_m800 (hidden information games)')
	parser.add_argument('--resume', action='store_true', help='skip what is already done, continue an interrupted stage')
	g = parser.add_argument_group('takeoff (1_takeoff)')
	g.add_argument('--takeoff-score', type=float, default=0.8, help='share of decisive arena games the new net must win')
	g.add_argument('--takeoff-score2', type=float, default=0.7, help='share of decisive games needed in TWO arenas')
	g.add_argument('--takeoff-decisive', type=int, default=20, help='minimum decisive games in such an arena')
	g.add_argument('--takeoff-extra', type=int, default=3, help='single iterations of 200 games after the first 2 x 100')
	g.add_argument('--takeoff-restarts', type=int, default=1, help='restarts from scratch without takeoff, then warn and go on')
	g = parser.add_argument_group('blocks (2_m100, 3_m800)')
	g.add_argument('--block', type=int, default=5, help='iterations per block')
	g.add_argument('--pit-games', type=int, default=200, help='games of each block-end pit')
	g.add_argument('--m100-threshold', type=float, default=0.57, help='block-end score needed to stay in m100')
	g.add_argument('--m100-blocks', type=int, default=6, help='maximum blocks in m100')
	g.add_argument('--m800-threshold', type=float, default=0.55, help='block-end score needed to go on in m800')
	g.add_argument('--m800-blocks', type=int, default=10,
	               help='maximum blocks in m800 (raise it with --resume to extend a run stopped on this limit)')
	args = parser.parse_args(argv)
	args.train_m800 = args.train_m800 or args.train

	bad = FORBIDDEN_COMMON.intersection(common)
	if bad:
		raise SystemExit(f'[scenario] {sorted(bad)} are set by the scenario, remove them from the args after --')
	first, last = STAGES.index(args.start_at), STAGES.index(args.until)
	if first > last:
		raise SystemExit('[scenario] --start-at comes after --until')
	if first > 0 and not args.load:
		raise SystemExit(f'[scenario] --start-at {args.start_at} needs --load')

	os.makedirs(args.checkpoint, exist_ok=True)
	rows = []
	net = args.load
	forget = bool(args.load) and not os.path.isfile(os.path.join(os.path.dirname(args.load) or '.', 'checkpoint.examples'))
	if first <= 0 <= last:
		net = run_takeoff(args, net, forget, rows, common)
	if first <= 1 <= last:
		net = run_blocks(args, 'm100', net, rows, common)
	if first <= 2 <= last:
		net = run_blocks(args, 'm800', net, rows, common)

	shutil.copyfile(net, os.path.join(args.checkpoint, 'final.pt'))
	print_summary(args, rows, net)


if __name__ == '__main__':
	main()
