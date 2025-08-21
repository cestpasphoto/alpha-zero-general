import numpy as np
from numba import njit
import numba

from .AkropolisConstants import *

############################## BOARD DESCRIPTION ##############################
#
# Game.state : np.ndarray[int8] of shape (CITY_SIZE, CITY_SIZE, 3*N_PLAYERS+2)
#
#   axis 0 → r                ∈ [0..CITY_SIZE-1]
#   axis 1 → q                ∈ [0..CITY_SIZE-1]
#   axis 2 → cities and misc  ∈ [0..3*N_PLAYERS+1]
#     z=0             → tile description for city of player 0 
#     z=1             → tile description for city of player 1
#     ...
#     z=N_PLAYERS     → tile height for city of player 0  (0 if empty)
#     ...
#     z=2*N_PLAYERS   → tile ID city of player 0
#     ...
#
#     z=3*N_PLAYERS   → per-player scalars packed. For position (r,q):
#       (0..N_PLAYERS-1            , 0..N_COLORS-1) → plazas[p, c], nb of valid plazas of color c for player p
#       (N_PLAYERS..2*N_PLAYERS-1  , 0..N_COLORS-1) → districts[p, c], nb of valid districts (weighted by height) of
#                                                     color c for player p
#       (2*N_PLAYERS..3*N_PLAYERS-1, 0)             → total_scores[p], current score of player p (ensure no overflow)
#       (2*N_PLAYERS..3*N_PLAYERS-1, 1)             → stones[p], current nb of stones for player p
#
#     z=3*N_PLAYERS+1 → global scalars packed. For position (r,q):
#       (0..CONSTR_SITE_SIZE-1, 0..3)               → construction_site[i][j] is the descr of the j-th hex of the i-th
#                                                     tile on the construction site (empty always at the end).
#                                                     construction_site[i][3] is the tile ID
#       (CONSTR_SITE_SIZE,0..PACKED_TILES_BYTES-1)  → tiles_bitpack, i-th bit is 1 if tile id i is available
#       (CONSTR_SITE_SIZE+1,0..1)                   → misc: 0 = round number, 1 = remaining number of stacks
#
# The grid uses an odd-r offset coordinate system (r, q) for each hex. See
# https://www.redblobgames.com/grids/hexagons/#coordinates q increases eastward
# and r increases southeastward. The origin is on the upper left, and all
# coordinates are positive. When we need a 1D representation, we use a single
# scalar 'idx' = r * CITY_SIZE + q
#
# Example of odd-r offset coordinates around (3,3):
#             (2, 3)      (2, 4)
#                \          /
#                 \        /
#       (3, 2) ---- (3, 3) ---- (3, 4)
#                 /        \
#                /          \
#             (4, 3)      (4, 4)
#
# Tile description is defined in AkropolisConstants.py
#
############################## ACTION DESCRIPTION #############################
#
# Each possible move (placing a tile with a given orientation on a given board position)
# is mapped to a unique integer ID in a fixed, global action space.
#
#     action_id = tile_idx_in_cs * (CONSTR_SITE_SIZE * N_ORIENTS)
#               + site_idx * N_ORIENTS
#               + orient_idx
# Where:
#   - tile_idx_in_cs ∈ [0, CONSTR_SITE_SIZE − 1] # index of the tile type being placed
#   - site_idx       ∈ [0, CITY_AREA − 1]        # 1D index of the center hexagon on the board
#   - orient_idx     ∈ [0, N_ORIENTS − 1]        # rotation/orientation index

@njit(cache=True, fastmath=True, nogil=True)
def observation_size():
	return (CITY_SIZE, CITY_SIZE, 3*N_PLAYERS+2)

@njit(cache=True, fastmath=True, nogil=True)
def action_size():
	return CONSTR_SITE_SIZE * CITY_AREA * N_ORIENTS

mask = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)

@njit(cache=True, fastmath=True, nogil=True)
def my_packbits(array):
	b = np.asarray(array, dtype=np.uint8)
	pad = (-b.size) % 8
	if pad:
		b = np.concatenate((b, np.zeros(pad, dtype=np.uint8)))
	b = b.reshape(-1, 8)

	packed = (b * mask).sum(axis=1).astype(np.int8)
	return packed

@njit(cache=True, fastmath=True, nogil=True)
def my_unpackbits(values):
	p = np.asarray(values, dtype=np.int8).astype(np.uint8)

	bits_matrix = (p[:, None] & mask) > 0
	bools = bits_matrix.reshape(-1)
	return bools

@njit(cache=True, fastmath=True, nogil=True, inline='always')
def rotate_cell(idx, k):
	if idx < 0:
		return -1
	# coord on odd-r grid → cube coords
	r = idx // CITY_SIZE
	q = idx - r * CITY_SIZE
	x = q - ((r - (r & 1)) // 2)
	z = r
	y = -x - z
	# k× rotation 60° CW
	for _ in range(k):
		x, y, z = -z, -x, -y
	# back to odd-r
	r2 = z
	q2 = x + ((r2 - (r2 & 1)) // 2)
	if 0 <= r2 < CITY_SIZE and 0 <= q2 < CITY_SIZE:
		return r2 * CITY_SIZE + q2
	else:
		return -1

@njit(cache=True, fastmath=True, nogil=True, inline='always')
def rotate_pattern(pat, k):
	# PATTERNS est un np.ndarray (N_PATTERNS×3)
	c0 = rotate_cell(PATTERNS[pat, 0], k)
	c1 = rotate_cell(PATTERNS[pat, 1], k)
	c2 = rotate_cell(PATTERNS[pat, 2], k)
	# recherche linéaire
	for j in range(N_PATTERNS):
		if (PATTERNS[j, 0] == c0 and
			PATTERNS[j, 1] == c1 and
			PATTERNS[j, 2] == c2):
			return j
	return -1


# --- Réflexion de cellule sur axe hexagonal (odd-r offset) ---
@njit(cache=True, fastmath=True, nogil=True, inline='always')
def reflect_cell(idx):
	if idx < 0:
		return -1
	# conversion odd-r → cube coords
	r = idx // CITY_SIZE
	q = idx - r * CITY_SIZE
	x = q - ((r - (r & 1)) // 2)
	z = r
	y = -x - z
	# réflexion de base : inversion y ↔ z
	y, z = z, y
	# conversion cube → odd-r
	r2 = z
	q2 = x + ((r2 - (r2 & 1)) // 2)
	if 0 <= r2 < CITY_SIZE and 0 <= q2 < CITY_SIZE:
		return r2 * CITY_SIZE + q2
	else:
		return -1

# --- Réflexion d'un pattern (identification de l'indice équivalent) ---
@njit(cache=True, fastmath=True, nogil=True, inline='always')
def reflect_pattern(pat):
	c0 = reflect_cell(PATTERNS[pat, 0])
	c1 = reflect_cell(PATTERNS[pat, 1])
	c2 = reflect_cell(PATTERNS[pat, 2])
	# recherche linéaire de l'indice de pattern réfléchi
	for j in range(N_PATTERNS):
		if (PATTERNS[j, 0] == c0 and
			PATTERNS[j, 1] == c1 and
			PATTERNS[j, 2] == c2):
			return j
	return -1

# --- Réflexion de l'état complet ---
@njit(cache=True, fastmath=True, nogil=True, inline='always')
def reflect_state(state):
	new_s = np.zeros_like(state)
	# appliquer réflexion cellule à cellule
	for r in range(CITY_SIZE):
		for q in range(CITY_SIZE):
			old = r * CITY_SIZE + q
			nb  = reflect_cell(old)
			if nb >= 0:
				r2 = nb // CITY_SIZE
				q2 = nb - r2 * CITY_SIZE
				new_s[r2, q2, :] = state[r, q, :]
	# conserver les couches misc (round, tiles_bitpack, etc.)
	new_s[:, :, 3*N_PLAYERS:] = state[:, :, 3*N_PLAYERS:].copy()
	return new_s


# Pre-compute the list of 6 neighbours of position idx
NEIGHBORS = np.full((CITY_AREA, 6), -1, dtype=np.int16)
for r in range(CITY_SIZE):
	for q in range(CITY_SIZE):
		idx = r * CITY_SIZE + q
		cnt = 0
		for dq, dr in DIRECTIONS_ODD if (r%2)==1 else DIRECTIONS_EVEN:
			nq, nr = q + dq, r + dr
			if 0 <= nq < CITY_SIZE and 0 <= nr < CITY_SIZE:
				NEIGHBORS[idx, cnt] = nr * CITY_SIZE + nq
				cnt += 1

# Pre-compute the list of positions of the 3 hexes of tile depending on
#   the position of its center (idx)
#   its orientation (orient)
# PATTERNS[idx*N_ORIENTS+orient, :] = (n1, s, n2) or (-1, -1, -1)
N_PATTERNS = CITY_AREA * N_ORIENTS
PATTERNS = np.full((N_PATTERNS, 3), -1, dtype=np.int16)
for s in range(CITY_AREA):
	r, q = divmod(s, CITY_SIZE)
	for o in range(N_ORIENTS):
		idx = s * N_ORIENTS + o
		if (r%2)==1:
			d1, d2 = DIRECTIONS_ODD[o] , DIRECTIONS_ODD[(o + 1) % N_ORIENTS]
		else:
			d1, d2 = DIRECTIONS_EVEN[o], DIRECTIONS_EVEN[(o + 1) % N_ORIENTS]
		pts = [
			(q + d1[0], r + d1[1]),
			(q        , r),
			(q + d2[0], r + d2[1]),
		]
		if all(0 <= qq < CITY_SIZE and 0 <= rr < CITY_SIZE for qq, rr in pts):
			for j, (qq, rr) in enumerate(pts):
				PATTERNS[idx, j] = rr * CITY_SIZE + qq

# Pre-compute the list of positions of the 9 (or less) neighbors of a tile
# Using same index system as above
PATTERN_NEI = np.full((N_PATTERNS, 9), -1, dtype=np.int16)
for p in range(N_PATTERNS):
	triplet = PATTERNS[p]
	neighbors_set = set()
	for cell in triplet:
		if cell < 0:
			continue
		for neighbor in NEIGHBORS[cell]:
			if neighbor >= 0 and neighbor not in triplet:
				neighbors_set.add(neighbor)
	PATTERN_NEI[p, :len(neighbors_set)] = sorted(neighbors_set)

@njit(cache=True, fastmath=True, nogil=True)
def encode_score_to_int8(score: np.int16) -> np.int8:
    """Encode score(s) into an np.int8.
    - 0..200 -> v = s - 128
    - 201..310 -> v = 73 + ((s - 201) // 2)
    """
    score_ = np.int16(score)
    if score_ < 0 or score_ > 310:
        raise ValueError("score_ must be in [0,300]")
    v = score_ - 128 if score_ <= 200 else 73 + ((score_ - 201) // 2)
    return np.int8(v)

@njit(cache=True, fastmath=True, nogil=True)
def decode_value_from_int8(stored: np.int8) -> np.int16:
    """Decode stored np.int8 value(s) back to score
    - stored <= 72  -> s = v + 128  (exact for 0..200)
    - stored >= 74  -> s = 202 + 2*(v - 74)  (upper representative of 2-value bin)
    """
    v = np.int16(stored)
    score = (v + 128) if v <= 72 else (202 + 2 * (v - 73))
    if score < 0:
    	score = np.int16(0)
    elif score > 310:
    	score = np.int16(310)
    return score

spec = [
	('state'            , numba.int8[:,:,:]),
	('board_descr'      , numba.int8[:,:,:]),
	('board_height'     , numba.int8[:,:,:]),
	('board_tileID'     , numba.int8[:,:,:]),
	('plazas'           , numba.int8[:,:]),
	('districts'        , numba.int8[:,:]),
	('total_scores'     , numba.int8[:]),
	('stones'           , numba.int8[:]),
	('construction_site', numba.int8[:,:]),
	('tiles_bitpack'    , numba.int8[:]),
	('misc'             , numba.int8[:]),
]
@numba.experimental.jitclass(spec)
class Board():
	def __init__(self, num_players):
		self.state = np.zeros((CITY_SIZE, CITY_SIZE, 3*N_PLAYERS+2), dtype=np.int8)
		self.init_game()

	def init_game(self):
		self.copy_state(np.zeros((CITY_SIZE, CITY_SIZE, 3*N_PLAYERS+2), dtype=np.int8), copy_or_not=False)
		self.stones[:] = np.arange(1, N_PLAYERS+1, dtype=np.int8)
		self.tiles_bitpack[:] = my_packbits(TILES_DATA[:,3] <= N_PLAYERS)
		self.misc[1] = N_STACKS
		self.total_scores[:] = [encode_score_to_int8(self.stones[i]) for i in range(N_PLAYERS)]
		# Set initial tile
		self.board_descr [START_TILE_R, START_TILE_Q, :] = PLAZA_BLUE
		self.board_height[START_TILE_R, START_TILE_Q, :] = 1
		self.board_tileID[START_TILE_R, START_TILE_Q, :] = TILES_DATA.shape[0]
		self.plazas[:,BLUE] = 1
		for idx in NEIGHBORS[START_TILE_R*CITY_SIZE+START_TILE_Q, ::2]:
			rr, qq = divmod(idx, CITY_SIZE)
			self.board_descr [rr, qq, :] = QUARRY
			self.board_height[rr, qq, :] = 1
			self.board_tileID[rr, qq, :] = TILES_DATA.shape[0]

		# Build construction site
		self.construction_site[:, :] = EMPTY
		self._draw_tiles_constr_site(random_seed=0, initial_draw=True)
		# All other items are zero

	def copy_state(self, state, copy_or_not):
		if self.state is state and not copy_or_not:
			return
		self.state = state.copy() if copy_or_not else state

		self.board_descr       = self.state[:, :,            :  N_PLAYERS]
		self.board_height      = self.state[:, :,   N_PLAYERS:2*N_PLAYERS]
		self.board_tileID      = self.state[:, :, 2*N_PLAYERS:3*N_PLAYERS]
		self.plazas            = self.state[           :  N_PLAYERS, :N_COLORS, 3*N_PLAYERS]
		self.districts         = self.state[  N_PLAYERS:2*N_PLAYERS, :N_COLORS, 3*N_PLAYERS]
		self.total_scores      = self.state[2*N_PLAYERS:3*N_PLAYERS, 0        , 3*N_PLAYERS]
		self.stones            = self.state[2*N_PLAYERS:3*N_PLAYERS, 1        , 3*N_PLAYERS]
		self.construction_site = self.state[0:CONSTR_SITE_SIZE, :4                 , 3*N_PLAYERS+1]
		self.tiles_bitpack     = self.state[CONSTR_SITE_SIZE  , :PACKED_TILES_BYTES, 3*N_PLAYERS+1]
		# misc contains round number (0) and remaining number of stacks (1)
		self.misc              = self.state[CONSTR_SITE_SIZE+1, :2                 , 3*N_PLAYERS+1]

	def make_move(self, move, player, random_seed):
		# decode "move" using divmod
		tile_idx_in_cs, rem  = divmod(move, N_PATTERNS)
		idx, orient          = divmod(rem, N_ORIENTS)

		# remove tile from construction_site, and move further tiles
		tile = self.construction_site[tile_idx_in_cs, :].copy()
		self.construction_site[tile_idx_in_cs:-1, :] = self.construction_site[tile_idx_in_cs+1:, :]
		self.construction_site[-1, :] = EMPTY
		for desc, hex_idx in zip(tile[:3], PATTERNS[idx*N_ORIENTS+orient]):
			rr, qq = divmod(hex_idx, CITY_SIZE)
			# update internals if building upon a quarry or plaza (district will be managed later)
			under_type, under_color = DESCR_TO_TYPE_COLOR[self.board_descr[rr, qq, player]]
			if under_type == PLAZA:
				self.plazas[player, under_color] += 1
			if under_type == QUARRY:
				self.stones[player] += 1
			self.board_descr [rr, qq, player] = desc
			self.board_height[rr, qq, player] += 1
			self.board_tileID[rr, qq, player] = tile[3]
			# Update plazas
			if DESCR_TO_TYPE_COLOR[desc][0] == PLAZA:
				self.plazas[player, DESCR_TO_TYPE_COLOR[desc][1]] += 1

		# update stones, districts, total_scores for current player
		self.stones[player] -= tile_idx_in_cs
		self._update_districts(player)
		total = (self.districts[player, :] * self.plazas[player, :] * PLAZA_STARS[:]).sum() + self.stones[player]
		self.total_scores[player] = encode_score_to_int8(total)

		# Round number
		self.misc[0] += 1

		# if only 1 tile remaining in construction_site, draw a new set of tiles
		if self.construction_site[1, 0] == EMPTY and self.misc[1] > 0:
			self._draw_tiles_constr_site(random_seed, initial_draw=False)
			self.misc[1] -= 1 # remaining number of stacks

		return (player+1)%N_PLAYERS

	def valid_moves(self, player):
		heights_flat = self.board_height[:, :, player].ravel()
		tilesID_flat = self.board_tileID[:, :, player].ravel()

		# Identify which patterns are valid placements (independent of the chosen tile)
		pattern_is_valid = np.zeros(N_PATTERNS, dtype=np.bool_)
		for pattern_id in range(N_PATTERNS):
			cell_a = PATTERNS[pattern_id, 0]
			# Skip patterns that go off the board
			if cell_a < 0:
				continue
			cell_b = PATTERNS[pattern_id, 1]
			cell_c = PATTERNS[pattern_id, 2]

			height_a = heights_flat[cell_a]
			height_b = heights_flat[cell_b]
			height_c = heights_flat[cell_c]

			# All three cells must have the same height
			if (height_a != height_b) or (height_a != height_c):
				continue

			# If placing on empty ground, ensure connectivity to existing stacks
			if height_a == 0:
				connected = False
				# Check all neighbors of the triple
				for neighbor_idx in PATTERN_NEI[pattern_id]:
					if neighbor_idx < 0:
						# End of neighbor list
						break
					if heights_flat[neighbor_idx] > 0:
						connected = True
						break
				if not connected:
					continue
			# Else check that building above more than a single tile
			else:
				tileID_a = tilesID_flat[cell_a]
				tileID_b = tilesID_flat[cell_b]
				tileID_c = tilesID_flat[cell_c]

				if (tileID_a == tileID_b) and (tileID_a == tileID_c):
					continue

			# Pattern is valid for placement
			pattern_is_valid[pattern_id] = True

		result = np.zeros(CONSTR_SITE_SIZE * N_PATTERNS, dtype=np.bool_)
		# For each available tile slot, apply the valid patterns
		for slot_index in range(min(self.stones[player]+1, CONSTR_SITE_SIZE)):
			if self.construction_site[slot_index, 0] == EMPTY:
				# Skip empty slots
				continue

			base_offset = slot_index * N_PATTERNS
			for pattern_id in range(N_PATTERNS):
				if pattern_is_valid[pattern_id]:
					result[base_offset + pattern_id] = True

		return result

	def get_state(self):
		return self.state

	def get_round(self):
		return self.misc[0]

	def get_score(self, player):
		return decode_value_from_int8(self.total_scores[player])

	def check_end_game(self, next_player):
		if (self.misc[1] <= 0 and self.construction_site[1, 0] == EMPTY):
			# total_scores may not be precise enough
			# need to recompute
			m = self.total_scores.max()
			if m > 73:
				print(f'WARNING, max score is too high {self.total_scores[:]+127}')
			single_winner = int((self.total_scores == m).sum()) == 1
			return np.where(self.total_scores == m, np.float32(1.0 if single_winner else 0.001), np.float32(-1.0))
		return np.zeros((N_PLAYERS,), dtype=np.float32)

	# if n=1, transform P0 to Pn, P1 to P0, ... and Pn to Pn-1
	# else do this action n times
	def swap_players(self, nb_swaps):
		# Ensure nb_swaps is within [0..N_PLAYERS-1]
		nb_swaps = nb_swaps % N_PLAYERS
		if nb_swaps == 0:
			return

		# Create temporary copy vectors
		tmp_board_descr  = np.empty_like(self.board_descr)
		tmp_board_height = np.empty_like(self.board_height)
		tmp_board_tileID = np.empty_like(self.board_tileID)
		tmp_plazas = np.empty_like(self.plazas)
		tmp_districts = np.empty_like(self.districts)
		tmp_scores = np.empty_like(self.total_scores)
		tmp_stones = np.empty_like(self.stones)

		# Roll data - np.roll not available on Numba
		for p in range(N_PLAYERS):
			tmp_board_descr[:, :, p] = self.board_descr[:, :, (p + nb_swaps) % N_PLAYERS]
			tmp_board_height[:, :, p] = self.board_height[:, :, (p + nb_swaps) % N_PLAYERS]
			tmp_board_tileID[:, :, p] = self.board_tileID[:, :, (p + nb_swaps) % N_PLAYERS]
			tmp_plazas[p, :] = self.plazas[(p + nb_swaps) % N_PLAYERS, :]
			tmp_districts[p, :] = self.districts[(p + nb_swaps) % N_PLAYERS, :]
			tmp_scores[p] = self.total_scores[(p + nb_swaps) % N_PLAYERS]
			tmp_stones[p] = self.stones[(p + nb_swaps) % N_PLAYERS]

		self.board_descr [:, :, :] = tmp_board_descr
		self.board_height[:, :, :] = tmp_board_height
		self.board_tileID[:, :, :] = tmp_board_tileID
		self.plazas[:, :] = tmp_plazas
		self.districts[:, :] = tmp_districts
		self.total_scores[:] = tmp_scores
		self.stones[:] = tmp_stones

	def get_symmetries(self, policy, valids):
		"""
		Retourne la liste des (state, policy, valids) pour
		les 6 rotations et 6 réflexions axiales (dièdre D₆).
		"""
		syms = []
		base_s, base_p, base_v = self.state, policy, valids

		# on parcours d'abord sans réflexion puis avec réflexion
		for do_reflect in (False, True):
			# appliquer réflexion sur state et mapper policy/valids
			if do_reflect:
				s_in = reflect_state(base_s)
				p_temp = np.zeros_like(base_p)
				v_temp = np.zeros_like(base_v)
				for a in range(base_p.size):
					cs = a // N_PATTERNS
					pt = a % N_PATTERNS
					rp = reflect_pattern(pt)
					ni = cs * N_PATTERNS + rp
					p_temp[ni] = base_p[a]
					v_temp[ni] = base_v[a]
				p_in, v_in = p_temp, v_temp
			else:
				s_in = base_s.copy()
				p_in = base_p.copy()
				v_in = base_v.copy()

			# appliquer les 6 rotations
			for k in range(N_ORIENTS):
				# rotation de l'état
				new_s = np.zeros_like(s_in)
				for r in range(CITY_SIZE):
					for q in range(CITY_SIZE):
						old = r * CITY_SIZE + q
						nb  = rotate_cell(old, k)
						if nb >= 0:
							r2 = nb // CITY_SIZE
							q2 = nb - r2 * CITY_SIZE
							new_s[r2, q2, :] = s_in[r, q, :]
				# conserver misc
				new_s[:, :, 3*N_PLAYERS:] = s_in[:, :, 3*N_PLAYERS:].copy()

				# remapper policy & valids
				new_p = np.zeros_like(p_in)
				new_v = np.zeros_like(v_in)
				for a in range(p_in.size):
					cs = a // N_PATTERNS
					pt = a % N_PATTERNS
					rp = rotate_pattern(pt, k)
					ni = cs * N_PATTERNS + rp
					new_p[ni] = p_in[a]
					new_v[ni] = v_in[a]

				syms.append((new_s, new_p, new_v))
		return syms

	def _draw_tiles_constr_site(self, random_seed, initial_draw=False):
		tiles_availability = my_unpackbits(self.tiles_bitpack)
		for i in range(0 if initial_draw else 1, CONSTR_SITE_SIZE):
			available_tiles = np.flatnonzero(tiles_availability)
			if initial_draw or random_seed == 0:
				tile_id = np.random.choice(available_tiles)
			else:
				# https://stackoverflow.com/questions/3062746/special-simple-random-number-generator
				# m=61, c=42, a=2013+1
				rnd_value = (2014 * (random_seed+np.int64(self.misc[0])) + 42) % 61
				tile_id = available_tiles[rnd_value%len(available_tiles)]

			self.construction_site[i, :3] = TILES_DATA[tile_id, :3]
			self.construction_site[i, 3] = tile_id
			tiles_availability[tile_id] = False
		self.tiles_bitpack[:] = my_packbits(tiles_availability)

	def _update_districts(self, player: int):
		# 0) Récupère desc + hauteur et met à plat
		desc2d = self.board_descr [:, :, player]
		h2d    = self.board_height[:, :, player]
		desc   = desc2d.ravel()
		h      = h2d.ravel()
		district = np.zeros(N_COLORS, dtype=np.int32)

		# 1) GREEN (Jardins)
		mask_green = (desc == DISTRICT_GREEN)
		district[GREEN] = h[mask_green].sum()

		# 2) YELLOW (Marchés isolés)
		mask_yellow = (desc == DISTRICT_YELLOW)
		yellow_idxs = np.nonzero(mask_yellow)[0]
		score_y = 0
		for idx in yellow_idxs:
			isolated = True
			# si un voisin est aussi un marché YELLOW, ce n'est pas isolé
			for nb in NEIGHBORS[idx]:
				if nb >= 0 and desc[nb] == DISTRICT_YELLOW:
					isolated = False
					break
			if isolated:
				score_y += h[idx]
		district[YELLOW] = score_y

		# 3) PURPLE (Temples entourés)
		mask_purple = (desc == DISTRICT_PURPLE)
		purple_idxs = np.nonzero(mask_purple)[0]
		score_p = 0
		for idx in purple_idxs:
			# ne considérer que les hex complétés (6 voisins)
			valid_nbs = [nb for nb in NEIGHBORS[idx] if nb >= 0]
			if len(valid_nbs) == 6:
				surrounded = True
				for nb in valid_nbs:
					if h[nb] == 0:
						surrounded = False
						break
				if surrounded:
					score_p += h[idx]
		district[PURPLE] = score_p

		# 5) RED (Caserne en périphérie réelle)
		# Using flood fill algorithm to list tiles connected to the border
		is_empty = desc == EMPTY
		outer_empty = np.zeros_like(is_empty)
		for idx in np.nonzero(is_empty)[0]:
			for nb in NEIGHBORS[idx]:
				if nb < 0:
					outer_empty[idx] = True
					break
		stack = [i for i in np.nonzero(outer_empty)[0]]
		for cur in stack:
			for nb in NEIGHBORS[cur]:
				if nb < 0 or outer_empty[nb] or not is_empty[nb]:
					continue
				outer_empty[nb] = True
				stack.append(nb)
		mask_red = (desc == DISTRICT_RED)
		red_touch = np.zeros_like(mask_red)
		for idx in np.nonzero(mask_red)[0]:
			for nb in NEIGHBORS[idx]:
				if nb < 0 or outer_empty[nb]:
					red_touch[idx] = True
					break
		district[RED] = h[red_touch].sum()

		# 6) BLUE (Maisons) : plus longue chaîne
		mask_blue = (desc == DISTRICT_BLUE)
		visited = np.zeros_like(mask_blue)
		max_chain = 0
		for start in np.nonzero(mask_blue)[0]:
			if visited[start]:
				continue
			chain = 0
			stack = [start]
			visited[start] = True
			while stack:
				cur = stack.pop()
				chain += h[cur]
				for nb in NEIGHBORS[cur]:
					if nb < 0 or visited[nb] or not mask_blue[nb]:
						continue
					visited[nb] = True
					stack.append(nb)
			max_chain = max(max_chain, chain)
		district[BLUE] = max_chain

		# 7) Enregistrement des scores
		self.districts[player, :] = district
