"""Aggregate and print the trade-usage counters written by CatanGame._TradeStats.

Usage:
    CATAN_TRADE_STATS=/tmp/trade_stats python main.py ...     # self-play writes the .npz files
    python CatanTradeStats.py /tmp/trade_stats [--top 15]

Everything is MCTS policy mass on full-search self-play positions (the ones
that reach getSymmetries), not sampled moves: it reads as "what the search
wants", before the temperature.
"""
import sys, glob, os
import numpy as np

try:
	from CatanConstants import *
	from CatanDisplay import set_to_str, phase_char
except ImportError:                                  # pragma: no cover
	from .CatanConstants import *
	from .CatanDisplay import set_to_str, phase_char


def load(folder):
	files = sorted(glob.glob(os.path.join(folder, 'trade_stats_*.npz')))
	if not files:
		sys.exit(f'no trade_stats_*.npz in {folder}')
	acc = None
	for f in files:
		z = np.load(f)
		d = {k: z[k] for k in z.files}
		acc = d if acc is None else {k: acc[k] + d[k] for k in acc}
	return acc, len(files)


def report(st, n_files, top=15):
	n_pos = st['n_pos']; tot = int(n_pos.sum())
	print(f'{n_files} process file(s), {tot} full-search positions\n')
	print('positions by phase:')
	for ph in range(N_PHASES):
		if n_pos[ph]:
			print(f'  {phase_char[ph]:18s} {int(n_pos[ph]):8d}  {n_pos[ph] / tot * 100:5.1f}%')

	legal, mass, argmax = st['main']
	print(f'\nMAIN phase, opening a player trade:')
	print(f'  legal in            {legal / max(n_pos[PHASE_MAIN], 1) * 100:5.1f}% of MAIN positions')
	print(f'  mean policy mass    {mass / max(legal, 1) * 100:5.1f}%  (when legal)')
	print(f'  argmax is opening   {argmax / max(legal, 1) * 100:5.1f}%  (when legal)')

	ok, no, counter = st['answer']
	s = ok + no + counter
	print(f'\nTRADE ANSWER ({int(n_pos[PHASE_TRADE_ANSWER])} positions):  '
	      f'OK {ok / max(s, 1e-9) * 100:5.1f}%   NO {no / max(s, 1e-9) * 100:5.1f}%   counter {counter / max(s, 1e-9) * 100:5.1f}%')
	if n_pos[PHASE_TRADE_ACCEPT]:
		acc = st['accept']
		print(f'TRADE ACCEPT ({int(n_pos[PHASE_TRADE_ACCEPT])} positions):  ' + '   '.join(
			f'{"refuse all" if t == 0 else f"take P+{t}"} {acc[t] / max(acc.sum(), 1e-9) * 100:5.1f}%' for t in range(N_PLAYERS)))

	def top_sets(v, label):
		order = np.argsort(-v)[:top]
		print(f'\n{label} (mass share):')
		for i in order:
			if v[i] > 0:
				print(f'  {set_to_str(TRADE_SETS[i]):10s} {v[i] / max(v.sum(), 1e-9) * 100:5.1f}%')
	top_sets(st['recv'], 'ASKED sets')
	top_sets(st['give'], 'OFFERED sets')

	pair = st['pair']; ap = st['answer_pair']
	flat = np.argsort(-pair, axis=None)[:top]
	print(f'\ntop (asked -> offered) announcements, with responders\' answer to that announcement:')
	print(f'  {"asked":10s} {"offered":10s} {"share":>6s}  {"OK":>6s} {"NO":>6s}')
	for f in flat:
		a, g = divmod(int(f), N_TRADE_SETS)
		if pair[a, g] <= 0:
			break
		okm, nom = ap[0, a, g], ap[1, a, g]
		rate = '' if okm + nom == 0 else f'{okm / (okm + nom) * 100:5.0f}% {nom / (okm + nom) * 100:5.0f}%'
		print(f'  {set_to_str(TRADE_SETS[a]):10s} {set_to_str(TRADE_SETS[g]):10s} {pair[a, g] / max(pair.sum(), 1e-9) * 100:5.1f}%  {rate}')

	# size structure: how many cards asked vs offered
	na = TRADE_SETS.sum(1)
	m = np.zeros((3, 3))
	for a in range(N_TRADE_SETS):
		for g in range(N_TRADE_SETS):
			m[na[a] - 1, na[g] - 1] += pair[a, g]
	print('\nsize of announcements (rows: cards asked 1..3, cols: cards offered 1..3), share:')
	for i in range(3):
		print('  ' + '  '.join(f'{m[i, j] / max(m.sum(), 1e-9) * 100:5.1f}%' for j in range(3)))


if __name__ == '__main__':
	folder = sys.argv[1] if len(sys.argv) > 1 else '.'
	top = int(sys.argv[sys.argv.index('--top') + 1]) if '--top' in sys.argv else 15
	st, n = load(folder)
	report(st, n, top)
