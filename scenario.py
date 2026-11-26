#!/usr/bin/env python3
"""Train a game from scratch with a fixed, game-independent staged recipe (v3.7).

Every folder lives directly in <checkpoint>/ (flat layout, one main.py run per folder):

  1_boot       m100, -q --q-boot, 100 games x 2 iterations, then up to --boot-extra single iterations of 200
               games until the arenas show a clear takeoff: one score >= --boot-score, or two scores
               >= --boot-score2, on >= --boot-decisive decisive games. Without takeoff, the folder is kept as
               1_boot_failedN and the stage restarts from scratch (--boot-restarts times), then goes on with a warning.
  2A_light,    m100, --games games per iteration, -q --q-light, one folder per block of --block iterations.
  2B_light...  After each block, pit (--pit-games games, m100) of the block end against the previous block end
               (or the boot net): next block while the score is >= --light-threshold, at most --light-blocks blocks.
  3A_deep...   m800, --games games per iteration, -q --q-deep, no forced playouts since v3.7 (--with-F to add -F:
               -61 Elo on Azul 2p, equal on Akropolis 3p), same blocks: next block while
               the block-end pit (m800) is >= --deep-threshold, at most --deep-blocks blocks. Raising --deep-blocks
               with --resume extends a run that stopped on the block limit.
  4_anneal     optional (--until anneal, off by default since v3.6: kept in 1 run out of 9): one more training
               pass at a lower learning rate (--anneal-train) on the buffer of the final deep net, without
               self-play; kept only if its pit (m800) against that net is >= --anneal-threshold.

--pit-openings passes --openings to every pit (deterministic games where pits replay few distinct games).

The net kept at the end of light / deep is the last block end, unless its pit against the previous block end is
below 0.5 (then the previous block end). The final net is copied to final.pt. One training config (--train,
--train-deep for the deep blocks). Inside a block, the arena gate of Coach stays the ratchet (-s is disabled,
the block-end pits decide when to stop).

Console labels: stage number, block letter (A = block 1), folder name, then what changes compared with the previous
stage (e.g. 2C_light_q1_e1000, 3A_deep_m800_F). Full parameters are in the command lines printed just before.
Every decision is stored in <checkpoint>/scenario.json, so --resume continues where it stopped.
Summary in <checkpoint>/scenario.tsv. Arena results come from the arenas.csv that Coach writes in each folder.

Example:
    ./scenario.py catan -V 13 -C ../results/v37_catan4 -- -u 2
    ./scenario.py catan -V 13 -C ../results/v37_catan4 --resume --until anneal   (adds the final pass to a finished run)
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

RECIPE = 'v3.7'
DEFAULT_TRAIN = '-b 32 -l 3e-3 -p 2'
STAGES = ['boot', 'light', 'deep', 'anneal']
STAGE_NUM = {'boot': 1, 'light': 2, 'deep': 3, 'anneal': 4}
FORBIDDEN_COMMON = {'-C', '--checkpoint', '-L', '--load-folder-file', '-V', '--nn-version', '-b', '--batch-size',
                    '-l', '--learn-rate', '-p', '--epochs', '-m', '--numMCTSSims', '--forget-examples',
                    '-e', '--numEps', '-s', '--stop-after-N-fail', '-n', '--numIters', '-q', '--q-weight', '-F',
                    '--forced-playouts'}
NO_STOP = '1000'   # -s value that never stops a block early
RE_PIT = re.compile(r'^RESULT\s+(\S+)\s+vs\s+(\S+)')
RE_SCORE = re.compile(r'score=([\d.]+)')


# ----------------------------------------------------------------------------- display

TTY = sys.stdout.isatty()


def color(text, code):
	return f'\033[{code}m{text}\033[0m' if TTY else text


TAG = color('[scenario]', '1;34')   # bold blue


def say(msg):
	print(f'{TAG} {msg}', flush=True)


def lab(text):
	return color(text, '1;36')       # bold cyan


def num(x):
	"""Compact number: 1.0 -> 1, 0.5 -> 0.5, 0.003 -> 3e-3."""
	x = float(x)
	if x == int(x):
		return str(int(x))
	if abs(x) < 0.01:
		mant, exp = f'{x:.0e}'.split('e')
		return f'{mant}e{int(exp)}'
	return f'{x:g}'


def ge(score, threshold):
	return '≥' if score >= threshold else '<'


# ----------------------------------------------------------------------------- helpers

def option_value(tokens, *names):
	for i, t in enumerate(tokens[:-1]):
		if t in names:
			return tokens[i + 1]
	return None


def elo(score):
	score = min(max(score, 1e-3), 1 - 1e-3)
	return 400 * math.log10(score / (1 - score))


def letter(k):
	"""Block letter: 0 -> A, 25 -> Z, 26 -> AA."""
	return chr(65 + k) if k < 26 else letter(k // 26 - 1) + chr(65 + k % 26)


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


def newest_net(folder):
	"""Current net of a folder: best.pt if an arena was ever accepted there (a rejection reloads temp.pt, which then
	holds the same weights), otherwise temp.pt (the starting net)."""
	best, temp = os.path.join(folder, 'best.pt'), os.path.join(folder, 'temp.pt')
	if os.path.isfile(best) and any(a[3] for a in read_arenas(folder)):
		return best
	return temp if os.path.isfile(temp) else (best if os.path.isfile(best) else None)


def stamp_part(folder, state):
	"""Rename the numbered checkpoints of the previous main.py run: a new run restarts at checkpoint_1."""
	part = state.get('parts', 0) + 1
	for f in glob.glob(os.path.join(folder, 'checkpoint_*.pt')):
		os.rename(f, os.path.join(folder, f'part{part}_' + os.path.basename(f)))
	state['parts'] = part


class Scenario:
	"""Run state, stored in <checkpoint>/scenario.json."""

	def __init__(self, args):
		self.args = args
		self.path = os.path.join(args.checkpoint, 'scenario.json')
		self.state = {}
		if os.path.isfile(self.path):
			with open(self.path) as f:
				self.state = json.load(f)
		self.state['recipe'] = RECIPE

	def save(self):
		os.makedirs(self.args.checkpoint, exist_ok=True)
		with open(self.path + '.tmp', 'w') as f:
			json.dump(self.state, f, indent=1)
		os.replace(self.path + '.tmp', self.path)

	def stage(self, name):
		return self.state.setdefault(name, {})

	def rel(self, path):
		"""Path relative to the run folder when it is inside it."""
		if path is None:
			return '-'
		r = os.path.relpath(path, self.args.checkpoint)
		return path if r.startswith('..') else r

	def folder(self, name):
		return os.path.join(self.args.checkpoint, name)


# ----------------------------------------------------------------------------- labels

def stage_params(args):
	"""What defines each stage, used to build the 'what changes' part of the labels."""
	def train_tokens(train):
		t = shlex.split(train)
		return dict(b=option_value(t, '-b', '--batch-size'), lr=option_value(t, '-l', '--learn-rate'),
		            p=option_value(t, '-p', '--epochs'))
	light = dict(m=100, q=args.q_light, e=args.games, F=False, **train_tokens(args.train))
	return {
		'boot': dict(m=100, q=args.q_boot, e='100/200', F=False, **train_tokens(args.train)),
		'light': light,
		'deep': dict(m=800, q=args.q_deep, e=args.games, F=not args.no_F, **train_tokens(args.train_deep)),
		'anneal': dict(light, m=800, q=args.q_deep, F=not args.no_F, **train_tokens(args.anneal_train)),
	}


def stage_label(args, stage):
	"""<num>_<name>_<what changes compared with the previous stage>, e.g. 3_deep_m800_F."""
	params = stage_params(args)
	cur = params[stage]
	if stage == 'boot':
		parts = [f'q{num(cur["q"])}']
	else:
		prev = params[STAGES[STAGES.index(stage) - 1]]
		parts = []
		for key in ('m', 'q', 'e', 'b', 'lr', 'p'):
			if stage == 'anneal' and key in ('m', 'q', 'e'):
				continue
			if str(cur[key]) != str(prev[key]):
				parts.append(f'{key}{num(cur[key]) if key in ("q", "lr") else cur[key]}')
		if stage != 'anneal' and cur['F'] != prev['F']:
			parts.append('F' if cur['F'] else 'noF')
	return '_'.join([f'{STAGE_NUM[stage]}_{stage}'] + parts)


def block_label(args, stage, k):
	"""Label of block k (0-based) of a stage: 2C_light_q1_e1000."""
	label = stage_label(args, stage)
	n = str(STAGE_NUM[stage])
	return n + letter(k) + label[len(n):]


def block_folder(stage, k):
	return f'{STAGE_NUM[stage]}{letter(k)}_{stage}'


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
	if args.pit_openings:
		cmd += ['--openings']
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

def next_stage_text(args, stage):
	"""'stage 3' if the next stage will run, 'done' otherwise."""
	i = STAGES.index(stage)
	return f'stage {i + 2}' if i + 1 <= STAGES.index(args.until) and i + 1 < len(STAGES) else 'done'


def takeoff(args, arenas):
	"""(1-based arena index, reason) of the takeoff, (None, None) if there is none yet: one arena with a share of
	decisive games >= --boot-score, or a second one >= --boot-score2 (each with >= --boot-decisive decisive games)."""
	n_second = 0
	for i, (new, old, _, _) in enumerate(arenas, start=1):
		if new + old < args.boot_decisive:
			continue
		share = new / (new + old)
		if share >= args.boot_score:
			return i, f'1 arena ≥ {args.boot_score:.2f}'
		if share >= args.boot_score2:
			n_second += 1
			if n_second >= 2:
				return i, f'2 arenas ≥ {args.boot_score2:.2f}'
	return None, None


def run_boot(args, scn, start, forget, rows, common):
	"""Short m100 runs until the arenas show that training took off; restart from scratch if it never does."""
	parts = [(100, 2)] + [(200, 1)] * args.boot_extra
	folder = scn.folder('1_boot')
	label = stage_label(args, 'boot')
	st = scn.stage('boot')
	if not st.get('done'):
		say(f'===== {lab(label)} · 2 it. × 100 games, then up to {args.boot_extra} it. × 200 · takeoff = '
		    f'1 arena ≥ {args.boot_score:.2f} or 2 arenas ≥ {args.boot_score2:.2f} (≥ {args.boot_decisive} decisive) =====')
	while not st.get('done'):
		if not st.get('parts_done') and not st.get('launched'):
			st.update(start=start, forget=forget, minutes=st.get('minutes', 0.), attempt=st.get('attempt', 0))
		while st.get('parts_done', 0) < len(parts):
			k = st.get('parts_done', 0)
			e, n = parts[k]
			if k == 0 and not st.get('launched'):
				load, fgt = st['start'], st['forget']
			else:
				stamp_part(folder, st)
				load, fgt = newest_net(folder), False
			st['launched'] = True
			scn.save()
			tokens = ['-m', '100', '-e', str(e), '-q', str(args.q_boot), '-n', str(n), '-s', NO_STOP]
			st['minutes'] += run_main(args, folder, tokens, args.train, load, fgt, common)
			st['parts_done'] = k + 1
			it, reason = takeoff(args, read_arenas(folder))
			part_label = f'1{letter(k)}' + label[1:]
			if it:
				st.update(takeoff_iter=it, done=True, final=newest_net(folder))
				scn.save()
				say(f'{lab(part_label)} · takeoff at it. {it} ({reason}) · {st["minutes"]:.0f} min → '
				    f'{next_stage_text(args, "boot")}, keeping {scn.rel(st["final"])}')
				break
			scn.save()
			if st['parts_done'] < len(parts):
				say(f'{lab(part_label)} → no takeoff yet')
		if st.get('done'):
			break
		n_it = sum(n for _, n in parts)
		if st['attempt'] < args.boot_restarts:
			attempt = st['attempt'] + 1
			failed = scn.folder(f'1_boot_failed{attempt}')
			say(f'{lab(f"1{letter(len(parts) - 1)}" + label[1:])} → no takeoff after {n_it} it., '
			    f'kept in {scn.rel(failed)}, restarting from scratch')
			os.rename(folder, failed)
			minutes = st['minutes']
			st.clear()
			st.update(start=start, forget=forget, minutes=minutes, attempt=attempt)
			scn.save()
		else:
			st.update(warning='NO TAKEOFF', done=True, final=newest_net(folder))
			scn.save()
			say(color('WARNING', '1;33') + f' · no takeoff after {args.boot_restarts + 1} attempts, going on anyway: '
			    'check the game logic / network, or lower --boot-score')
	rows.append(dict(step=label, games='100/200', m=100, q=num(args.q_boot), arenas=arena_summary(read_arenas(folder)),
	                 minutes=st['minutes'], pit_minutes=0, score=None,
	                 decision=st.get('warning') or f'takeoff it.{st.get("takeoff_iter")}'))
	write_tsv(args, rows)
	return st['final']


def run_block(args, scn, stage, folder, m, q, load, common):
	"""One block of --block iterations in its own folder, resumed if it was interrupted."""
	bstate_path = os.path.join(folder, 'block.json')
	bstate = {}
	if os.path.isfile(bstate_path):
		with open(bstate_path) as f:
			bstate = json.load(f)

	def save_block():
		os.makedirs(folder, exist_ok=True)
		with open(bstate_path, 'w') as f:
			json.dump(bstate, f, indent=1)

	tokens = ['-m', str(m), '-e', str(args.games), '-q', str(q), '-s', NO_STOP] + (['-F'] if m == 800 and not args.no_F else [])
	train = args.train_deep if stage == 'deep' else args.train
	done = len(read_arenas(folder))
	while done < args.block:
		if bstate.get('launched'):
			stamp_part(folder, bstate)
			load = newest_net(folder) or load
		bstate['launched'] = True
		save_block()
		bstate['minutes'] = bstate.get('minutes', 0) + run_main(args, folder, tokens + ['-n', str(args.block - done)],
		                                                         train, load, False, common)
		save_block()
		done = len(read_arenas(folder))
	return bstate.get('minutes', 0)


def run_blocks(args, scn, stage, start, rows, common):
	"""Blocks of iterations until the block-end pit against the previous block end falls below the threshold."""
	m, q = (100, args.q_light) if stage == 'light' else (800, args.q_deep)
	threshold, max_blocks = (args.light_threshold, args.light_blocks) if stage == 'light' else (args.deep_threshold, args.deep_blocks)
	label = stage_label(args, stage)
	st = scn.stage(stage)
	st.setdefault('start', start)
	st.setdefault('blocks', [])
	scn.save()
	if st.get('done') and st.get('reason') == 'block limit' and len(st['blocks']) < max_blocks:
		say(f'{lab(label)} · extending: {len(st["blocks"])} block{"s" if len(st["blocks"]) > 1 else ""} done (block limit), limit raised to {max_blocks}')
		st['done'] = False
		if stage == 'deep' and scn.state.pop('anneal', None):   # the final pass must be redone on the new deep net
			stale = scn.folder('4_anneal')
			if os.path.isdir(stale):
				os.rename(stale, stale + f'_old{int(time.time())}')
	if not st.get('done'):
		say(f'===== {lab(label)} · blocks of {args.block} it. · next block while pit ≥ {threshold} · max {max_blocks} block{"s" if max_blocks > 1 else ""} =====')
	while not st.get('done'):
		k = len(st['blocks'])
		previous = st['blocks'][-1]['net'] if st['blocks'] else st['start']
		bfolder = scn.folder(block_folder(stage, k))
		minutes = run_block(args, scn, stage, bfolder, m, q, previous, common)
		net = newest_net(bfolder)
		t0 = time.time()
		score = pit_score(args, net, previous, m, args.pit_games, common)
		block = dict(folder=bfolder, net=net, previous=previous, minutes=minutes, score=score,
		             pit_minutes=(time.time() - t0) / 60, arenas=arena_summary(read_arenas(bfolder)))
		st['blocks'].append(block)
		if score < threshold:
			st['done'], st['reason'] = True, f'pit {score:.3f} < {threshold}'
		elif len(st['blocks']) >= max_blocks:
			st['done'], st['reason'] = True, 'block limit'
		if st.get('done'):
			st['final'] = net if score >= 0.5 else previous
			nxt = (f'{next_stage_text(args, stage)}{" (block limit)" if st["reason"] == "block limit" else ""}, '
			       f'keeping {scn.rel(st["final"])}{" (last block < 0.5)" if score < 0.5 else ""}')
		else:
			nxt = lab(block_label(args, stage, k + 1))
		scn.save()
		say(f'{lab(block_label(args, stage, k))} · {score:.3f} ({elo(score):+.0f} Elo) {ge(score, threshold)} {threshold} · '
		    f'arenas {block["arenas"]} · {minutes:.0f} min + pit {block["pit_minutes"]:.0f} min → {nxt}')
	n_blocks = len(st['blocks'])
	for i, b in enumerate(st['blocks']):
		if i < n_blocks - 1:
			decision = 'next'
		else:
			decision = f'→ {next_stage_text(args, stage)}' + (' (limit)' if st['reason'] == 'block limit' else '')
		rows.append(dict(step=block_label(args, stage, i), games=str(args.games), m=m, q=num(q), arenas=b['arenas'],
		                 minutes=b['minutes'], pit_minutes=b.get('pit_minutes', 0), score=b['score'], decision=decision))
	write_tsv(args, rows)
	return st['final']


def run_anneal(args, scn, net, rows, common):
	"""One extra training pass at a lower learning rate on the buffer of the final net, kept only if it wins its pit."""
	label = stage_label(args, 'anneal')
	folder = scn.folder('4_anneal')
	st = scn.stage('anneal')
	examples = os.path.join(os.path.dirname(net), 'checkpoint.examples')
	out = os.path.join(folder, 'anneal.pt')
	if not st.get('done'):
		say(f'===== {lab(label)} · 1 training pass on {scn.rel(examples)} · kept if pit ≥ {args.anneal_threshold} =====')
		os.makedirs(folder, exist_ok=True)
		t0 = time.time()
		if not os.path.isfile(out):
			cmd = [sys.executable, os.path.join(HERE, 'train_on_examples.py'), args.game, net, examples, out] + shlex.split(args.anneal_train)
			say('$ ' + shlex.join(cmd))
			if subprocess.run(cmd, cwd=HERE).returncode != 0 or not os.path.isfile(out):
				raise SystemExit('[scenario] train_on_examples.py failed, stopping (rerun with --resume)')
		minutes = (time.time() - t0) / 60
		t0 = time.time()
		score = pit_score(args, out, net, 800, args.pit_games, common)
		kept = score >= args.anneal_threshold
		st.update(done=True, start=net, score=score, minutes=minutes, pit_minutes=(time.time() - t0) / 60,
		          final=out if kept else net)
		scn.save()
		say(f'{lab(label)} · {score:.3f} ({elo(score):+.0f} Elo) {ge(score, args.anneal_threshold)} {args.anneal_threshold} · '
		    f'{minutes:.0f} min + pit {st["pit_minutes"]:.0f} min → ' + ('kept' if kept else f'dropped, keeping {scn.rel(net)}'))
	rows.append(dict(step=label, games='-', m=800, q='-', arenas='-', minutes=st['minutes'], pit_minutes=st['pit_minutes'],
	                 score=st['score'], decision='kept' if st['final'] != st['start'] else 'dropped'))
	write_tsv(args, rows)
	return st['final']


# ----------------------------------------------------------------------------- report

COLUMNS = ['step', 'games', 'm', 'q', 'arenas', 'train', 'pit', 'score', 'elo', 'decision']


def row_cells(r):
	return [r['step'], str(r['games']), str(r['m']), str(r['q']), r['arenas'], f'{r["minutes"]:.0f}',
	        f'{r["pit_minutes"]:.0f}' if r['pit_minutes'] else '-',
	        '-' if r['score'] is None else f'{r["score"]:.3f}',
	        '-' if r['score'] is None else f'{elo(r["score"]):+.0f}', r['decision']]


def write_tsv(args, rows):
	with open(os.path.join(args.checkpoint, 'scenario.tsv'), 'w') as f:
		f.write(f'# recipe {RECIPE}\n')
		f.write('\t'.join(COLUMNS) + '\n')
		for r in rows:
			f.write('\t'.join(row_cells(r)) + '\n')


def print_summary(args, scn, rows, final):
	say(f'summary · recipe {RECIPE} · {args.game} -V {args.nn_version} · {args.checkpoint}')
	width = max([len(r['step']) for r in rows] + [len('step')])
	fmt = f'{{:<{width}}} {{:>7}} {{:>4}} {{:>4}} {{:>6}} {{:>6}} {{:>5}} {{:>6}} {{:>6}}  {{}}'
	print(fmt.format(*COLUMNS))
	for r in rows:
		print(fmt.format(*row_cells(r)))
	train_h, pit_h = sum(r['minutes'] for r in rows) / 60, sum(r['pit_minutes'] for r in rows) / 60
	print(f'TOTAL TIME: {train_h + pit_h:.1f} h (training {train_h:.1f} h + pits {pit_h:.1f} h)')
	print(f'final network: {scn.rel(final)} → {os.path.join(args.checkpoint, "final.pt")}', flush=True)


def resume_text(args, scn):
	"""'1_boot, 2A–2C, 3A done; 3B interrupted after 2/5 it.' from scenario.json and the block folders."""
	done = []
	if scn.state.get('boot', {}).get('done'):
		done.append('1_boot')
	interrupted = ''
	for stage in ('light', 'deep'):
		blocks = scn.state.get(stage, {}).get('blocks', [])
		n = STAGE_NUM[stage]
		if blocks:
			done.append(f'{n}A' if len(blocks) == 1 else f'{n}A–{n}{letter(len(blocks) - 1)}')
		nxt = scn.folder(block_folder(stage, len(blocks)))
		if not interrupted and os.path.isdir(nxt) and read_arenas(nxt):
			interrupted = f'; {n}{letter(len(blocks))} interrupted after {len(read_arenas(nxt))}/{args.block} it.'
	if scn.state.get('anneal', {}).get('done'):
		done.append('4_anneal')
	return (', '.join(done) + ' done' if done else 'nothing done') + interrupted


# ----------------------------------------------------------------------------- main

def main():
	argv = sys.argv[1:]
	common = []
	if '--' in argv:
		common, argv = argv[argv.index('--') + 1:], argv[:argv.index('--')]
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument('game')
	parser.add_argument('--nn-version', '-V', required=True, type=int, help='network version (mandatory)')
	parser.add_argument('--checkpoint', '-C', required=True, help='scenario folder (flat: one sub-folder per step)')
	parser.add_argument('--load', '-L', default=None,
	                    help='warm-start checkpoint for the first stage (its checkpoint.examples is used if present)')
	parser.add_argument('--start-at', choices=STAGES, default='boot', help='first stage to run (light/deep/anneal need --load)')
	parser.add_argument('--until', choices=STAGES, default=None,
	                    help='last stage to run (default: deep; anneal when --start-at anneal)')
	parser.add_argument('--train', default=DEFAULT_TRAIN, help=f'training config of boot and light (default: {DEFAULT_TRAIN})')
	parser.add_argument('--train-deep', default=None, help='training config of deep (default: same as --train)')
	parser.add_argument('--anneal-train', default='-b 32 -l 3e-4 -p 2', help='training config of the final pass (4_anneal)')
	parser.add_argument('--anneal-threshold', type=float, default=0.55, help='pit score needed to keep the final pass')
	parser.add_argument('--games', type=int, default=1000, help='games per iteration after boot')
	parser.add_argument('--q-boot', type=float, default=0., help='-q of boot (value target = z only while the net is random)')
	parser.add_argument('--q-light', type=float, default=1.0, help='-q of light')
	parser.add_argument('--q-deep', type=float, default=1.0, help='-q of deep')
	parser.add_argument('--with-F', action='store_true', help='forced playouts + policy target pruning (-F) in deep (off since v3.7)')
	parser.add_argument('--no-F', action='store_true', help='no forced playouts in deep: the default since v3.7, kept for old command lines')
	parser.add_argument('--resume', action='store_true', help='skip what is already done, continue an interrupted stage')
	g = parser.add_argument_group('boot (1_boot)')
	g.add_argument('--boot-score', type=float, default=0.8, help='share of decisive arena games the new net must win')
	g.add_argument('--boot-score2', type=float, default=0.7, help='share of decisive games needed in TWO arenas')
	g.add_argument('--boot-decisive', type=int, default=20, help='minimum decisive games in such an arena')
	g.add_argument('--boot-extra', type=int, default=3, help='single iterations of 200 games after the first 2 x 100')
	g.add_argument('--boot-restarts', type=int, default=1, help='restarts from scratch without takeoff, then warn and go on')
	g = parser.add_argument_group('blocks (light, deep)')
	g.add_argument('--block', type=int, default=5, help='iterations per block')
	g.add_argument('--pit-games', type=int, default=200, help='games of each block-end pit (and of the anneal pit)')
	g.add_argument('--pit-openings', action='store_true',
	               help='pits with drawn openings (pit.py --openings), for deterministic games such as abalone')
	g.add_argument('--light-threshold', type=float, default=0.57, help='block-end score needed to stay in light')
	g.add_argument('--light-blocks', type=int, default=6, help='maximum blocks in light')
	g.add_argument('--deep-threshold', type=float, default=0.55, help='block-end score needed to go on in deep')
	g.add_argument('--deep-blocks', type=int, default=10,
	               help='maximum blocks in deep (raise it with --resume to extend a run stopped on this limit)')
	args = parser.parse_args(argv)
	args.train_deep = args.train_deep or args.train
	if args.with_F and args.no_F:
		raise SystemExit('[scenario] --with-F and --no-F are exclusive')
	args.no_F = not args.with_F
	args.until = args.until or ('anneal' if args.start_at == 'anneal' else 'deep')

	bad = FORBIDDEN_COMMON.intersection(common)
	if bad:
		raise SystemExit(f'[scenario] {sorted(bad)} are set by the scenario, remove them from the args after --')
	first, last = STAGES.index(args.start_at), STAGES.index(args.until)
	if first > last:
		raise SystemExit('[scenario] --start-at comes after --until')
	if first > 0 and not args.load:
		raise SystemExit(f'[scenario] --start-at {args.start_at} needs --load')
	scn_path = os.path.join(args.checkpoint, 'scenario.json')
	if os.path.isdir(args.checkpoint) and os.listdir(args.checkpoint) and not args.resume:
		raise SystemExit(f'[scenario] {args.checkpoint} is not empty: use --resume or another -C')

	os.makedirs(args.checkpoint, exist_ok=True)
	resumed = os.path.isfile(scn_path)
	scn = Scenario(args)
	if resumed:
		say(f'recipe {RECIPE} · resuming from scenario.json: {resume_text(args, scn)}')
	elif first > 0:
		say(f'recipe {RECIPE} · start at stage {first + 1} from {args.load}')
	else:
		say(f'recipe {RECIPE}')
	scn.save()

	rows = []
	net = args.load
	forget = bool(args.load) and not os.path.isfile(os.path.join(os.path.dirname(args.load) or '.', 'checkpoint.examples'))
	if first <= 0 <= last:
		net = run_boot(args, scn, net, forget, rows, common)
	if first <= 1 <= last:
		net = run_blocks(args, scn, 'light', net, rows, common)
	if first <= 2 <= last:
		net = run_blocks(args, scn, 'deep', net, rows, common)
	if first <= 3 <= last:
		net = run_anneal(args, scn, net, rows, common)

	shutil.copyfile(net, os.path.join(args.checkpoint, 'final.pt'))
	print_summary(args, scn, rows, net)


if __name__ == '__main__':
	main()
