"""Aggregate and print the trade-usage counters written by CatanGame._TradeStats.

Usage:
    CATAN_TRADE_STATS=/tmp/trade_stats python main.py ...     # self-play writes the .npz files
    python CatanTradeStats.py /tmp/trade_stats [--top 15]

Two kinds of number live here, and mixing them up is the main way to read this
report wrong:
  - MASS  : MCTS policy mass, i.e. what the search WANTS (visit distribution at
            the root, after PTP, before Coach's sampling temperature).
  - PLAYED: what Coach actually sampled and played, on those very positions.
Everything is measured on full-search self-play positions only (the ones that
reach getSymmetries), so every ratio between two counters below is unbiased by
Playout Cap Randomization -- but no counter is an absolute per-game count until
it is rescaled by the setup phase (see "game accounting").
"""
import sys, glob, os
import numpy as np

try:
	from CatanConstants import *
	from CatanDisplay import set_to_str, phase_char
except ImportError:                                  # pragma: no cover
	from .CatanConstants import *
	from .CatanDisplay import set_to_str, phase_char

KEYS = ('n_pos', 'main', 'recv', 'give', 'pair', 'answer', 'answer_pair', 'accept',
        'legal', 'ent', 'hand', 'round_hist', 'played', 'recv_legal', 'give_legal')


def load(folder):
	files = sorted(glob.glob(os.path.join(folder, 'trade_stats_*.npz')))
	if not files:
		sys.exit(f'no trade_stats_*.npz in {folder}')
	acc = None
	for f in files:
		z = np.load(f)
		missing = [k for k in KEYS if k not in z.files]
		if missing:
			sys.exit(f'{f} was written by an older CatanGame (missing {missing}); delete the folder and re-run')
		d = {k: z[k] for k in KEYS}
		acc = d if acc is None else {k: acc[k] + d[k] for k in acc}
	return acc, len(files)


def report(st, n_files, top=15):
	n_pos = st['n_pos'].astype(np.float64); tot = float(n_pos.sum())
	played = st['played']; legal = st['legal']; ent = st['ent']; hand = st['hand']
	print(f'{n_files} process file(s), {int(tot)} full-search positions\n')

	# ---- game accounting: the setup phase is a fixed 2*P settlements + 2*P roads
	# per game, so it converts "positions seen" into "decisions per game" without
	# knowing the PCR ratio (it cancels out). Roads can be auto-resolved when only
	# one is legal, so the settlement estimate is the trustworthy one.
	ns, nr = n_pos[PHASE_SETUP_SETTLEMENT], n_pos[PHASE_SETUP_ROAD]
	print('game accounting (rescaled by the setup phase, PCR-independent):')
	if ns:
		print(f'  decisions per game   {tot * 2 * N_PLAYERS / ns:7.1f}   (+/- {tot * 2 * N_PLAYERS / ns / np.sqrt(ns):.0f}, from the {int(ns)} setup settlements)')
		print(f'                       {tot * 2 * N_PLAYERS / max(nr, 1):7.1f}   (same, from the {int(nr)} setup roads -- lower means some were forced)')
		print(f'  games in this sample {ns / (2 * N_PLAYERS):7.1f} x 1/p_full  (p_full = PCR full-search ratio)')
		for ph in range(N_PHASES):
			if n_pos[ph]:
				print(f'    {phase_char[ph]:18s} {n_pos[ph] * 2 * N_PLAYERS / ns:7.2f} per game')

	# ---- game length
	rh = st['round_hist'].astype(np.float64)
	nzr = np.flatnonzero(rh)
	if len(nzr):
		r0 = max(int(nzr[0]), 1)
		base = rh[r0:r0 + 3].mean() or 1.0
		surv = rh / base                      # positions per round ~ games still alive (x decisions/round)
		qs = []
		for frac in (0.5, 0.1, 0.01):
			idx = np.flatnonzero(surv >= frac)
			qs.append(int(idx[-1]) if len(idx) else 0)
		print(f'\ngame length: last round with >=50% / >=10% / >=1% of the early position rate: '
		      f'{qs[0]} / {qs[1]} / {qs[2]} rounds;  max seen {int(nzr[-1])};  '
		      f'{rh[MAX_ROUNDS:].sum() / max(rh.sum(), 1) * 100:.2f}% of positions at the MAX_ROUNDS guard')
		print('  (a proxy: positions per round, not games -- it also moves if decisions/round changes late)')

	# ---- per phase: how wide, how decided, and what got played
	print('\nper phase:  positions, mean legal moves, mean policy entropy, mean max(pi), mean hand size')
	for ph in range(N_PHASES):
		if not n_pos[ph]:
			continue
		print(f'  {phase_char[ph]:18s} {int(n_pos[ph]):6d}  legal {legal[ph] / n_pos[ph]:6.1f}   '
		      f'ent {ent[ph, 0] / n_pos[ph]:5.2f}   max(pi) {ent[ph, 1] / n_pos[ph] * 100:5.1f}%   hand {hand[ph] / n_pos[ph]:4.1f}')

	print('\nmoves PLAYED, per phase (share of the moves played in that phase):')
	for ph in range(N_PHASES):
		s = played[ph].sum()
		if not s:
			continue
		order = np.argsort(-played[ph])
		items = [f'{ACTION_BLOCKS[i][0]} {played[ph, i] / s * 100:.0f}%' for i in order if played[ph, i]]
		print(f'  {phase_char[ph]:18s} {int(s):6d}  ' + ', '.join(items[:6]))

	# ---- the opening decision: wanted vs played
	legal_n, mass, argmax, legal_ids, legal_moves = st['main']
	i_open = next(i for i, b in enumerate(ACTION_BLOCKS) if b[1] == A_TRADE_RECV)
	played_main = played[PHASE_MAIN].sum()
	print(f'\nMAIN phase, opening a player trade:')
	print(f'  legal in            {legal_n / max(n_pos[PHASE_MAIN], 1) * 100:5.1f}% of MAIN positions '
	      f'({legal_ids / max(legal_n, 1):.0f} opening ids legal there, out of {legal_moves / max(legal_n, 1):.0f} legal moves)')
	print(f'  uniform baseline    {legal_ids / max(legal_moves, 1e-9) * 100:5.1f}%  <- mass a random policy would put there')
	print(f'  mean policy mass    {mass / max(legal_n, 1) * 100:5.1f}%  (when legal)')
	print(f'  argmax is opening   {argmax / max(legal_n, 1) * 100:5.1f}%  (when legal)')
	print(f'  actually PLAYED     {played[PHASE_MAIN, i_open] / max(legal_n, 1) * 100:5.1f}%  (when legal)')
	print(f'  (Coach\'s sampling temperature can only move PLAYED between argmax and mass:')
	print(f'   anything outside that bracket comes from somewhere else.)')

	# ---- how many announcements never reach a GIVE ply (single legal giveaway -> auto-resolved)
	opened = played[PHASE_MAIN, i_open]
	give_plies = n_pos[PHASE_TRADE_OFFER]
	if opened:
		print(f'\nannouncements: {int(opened)} openings played, {int(give_plies)} GIVE plies seen '
		      f'-> {max(0.0, 1 - give_plies / opened) * 100:.0f}% of the GIVE plies were forced (single legal giveaway, auto-resolved)')
		print(f'  responders: {int(n_pos[PHASE_TRADE_ANSWER])} answer plies for {int(opened)} announcements x {N_PLAYERS - 1} responders '
		      f'-> {n_pos[PHASE_TRADE_ANSWER] / max(opened * (N_PLAYERS - 1), 1) * 100:.0f}% could pay what was asked (the rest is an auto-NO)')

	ok, no, counter = st['answer']
	s = ok + no + counter
	print(f'\nTRADE ANSWER ({int(n_pos[PHASE_TRADE_ANSWER])} positions):  '
	      f'OK {ok / max(s, 1e-9) * 100:5.1f}%   NO {no / max(s, 1e-9) * 100:5.1f}%   counter {counter / max(s, 1e-9) * 100:5.1f}%   (mass)')
	if played[PHASE_TRADE_ANSWER].sum():
		pa = played[PHASE_TRADE_ANSWER]
		i_ok = next(i for i, b in enumerate(ACTION_BLOCKS) if b[1] == A_TRADE_OK)
		spent = opened + n_pos[PHASE_TRADE_OFFER] + n_pos[PHASE_TRADE_ANSWER]
		k = 2 * N_PLAYERS / max(n_pos[PHASE_SETUP_SETTLEMENT], 1)
		print(f'  played: OK {pa[i_ok] / pa.sum() * 100:5.1f}%  -> {pa[i_ok] * k:.1f} trades closed per game, '
		      f'{spent * k:.1f} decisions spent on negotiation ({spent / max(tot, 1) * 100:.0f}% of all decisions, '
		      f'{spent / max(pa[i_ok], 1):.1f} per trade closed)')
	if n_pos[PHASE_TRADE_ACCEPT]:
		acc = st['accept']
		print(f'TRADE ACCEPT ({int(n_pos[PHASE_TRADE_ACCEPT])} positions):  ' + '   '.join(
			f'{"refuse all" if t == 0 else f"take P+{t}"} {acc[t] / max(acc.sum(), 1e-9) * 100:5.1f}%' for t in range(N_PLAYERS)))

	def top_sets(v, av, label):
		# `av` is how often the set was legal at all: mass alone cannot tell a
		# preference from an availability effect.
		share, base = v / max(v.sum(), 1e-9), av / max(av.sum(), 1e-9)
		order = np.argsort(-v)[:top]
		print(f'\n{label}:  mass share, availability share, ratio (>1 = preferred over what was on offer)')
		for i in order:
			if v[i] > 0:
				print(f'  {set_to_str(TRADE_SETS[i]):10s} {share[i] * 100:5.1f}%  {base[i] * 100:5.1f}%  '
				      f'{share[i] / base[i] if base[i] > 0 else float("nan"):5.2f}')
	top_sets(st['recv'], st['recv_legal'], 'ASKED sets')
	top_sets(st['give'], st['give_legal'], 'OFFERED sets')

	# ---- announcements as they were really made: read off the board at the ANSWER
	# ply, so both sides are the played ones (weighted by how many responders were
	# not auto-resolved, hence slightly biased towards affordable asks).
	ap = st['answer_pair']; real = ap[0] + ap[1]
	print(f'\ntop announcements REALLY made (asked -> offered), and how responders answered:')
	print(f'  {"asked":10s} {"offered":10s} {"share":>6s}  {"OK":>6s} {"NO":>6s}')
	for f in np.argsort(-real, axis=None)[:top]:
		a, g = divmod(int(f), N_TRADE_SETS)
		if real[a, g] <= 0:
			break
		print(f'  {set_to_str(TRADE_SETS[a]):10s} {set_to_str(TRADE_SETS[g]):10s} {real[a, g] / real.sum() * 100:5.1f}%  '
		      f'{ap[0, a, g] / real[a, g] * 100:5.0f}% {ap[1, a, g] / real[a, g] * 100:5.0f}%')

	# ---- size structure: what the search wants to announce, and what gets accepted
	na = TRADE_SETS.sum(1)
	pair = st['pair']
	want = np.zeros((3, 3)); done = np.zeros((3, 3)); okm = np.zeros((3, 3)); avail = np.zeros(3)
	for a in range(N_TRADE_SETS):
		avail[na[a] - 1] += st['give_legal'][a]
		for g in range(N_TRADE_SETS):
			want[na[a] - 1, na[g] - 1] += pair[a, g]
			done[na[a] - 1, na[g] - 1] += real[a, g]
			okm[na[a] - 1, na[g] - 1] += ap[0, a, g]
	print('\ngiveaways of 1/2/3 cards were legal ' + ' / '.join(f'{x / max(avail.sum(), 1e-9) * 100:.0f}%' for x in avail)
	      + ' of the time: the columns below read against that')
	print('announcement size (rows: cards asked 1..3, cols: cards offered 1..3)')
	print('  share of the GIVE-ply mass:          OK rate of the announcements really made:')
	for i in range(3):
		left = '  '.join(f'{want[i, j] / max(want.sum(), 1e-9) * 100:5.1f}%' for j in range(3))
		right = '  '.join((f'{okm[i, j] / done[i, j] * 100:5.0f}%' if done[i, j] > 0 else '    - ') for j in range(3))
		print(f'  {left}      {right}')


if __name__ == '__main__':
	folder = sys.argv[1] if len(sys.argv) > 1 else '.'
	top = int(sys.argv[sys.argv.index('--top') + 1]) if '--top' in sys.argv else 15
	st, n = load(folder)
	report(st, n, top)
