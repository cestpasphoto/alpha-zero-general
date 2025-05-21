import numpy as np
from .AkropolisConstants import *

############################## BOARD DESCRIPTION ##############################
#
# Game.state : np.ndarray[int8] of shape (CITY_SIZE, CITY_SIZE + OVERHEAD, N_PLAYERS, 2)
#
#   axis 0 → r                ∈ [0..CITY_SIZE-1]
#   axis 1 → q                ∈ [0..CITY_SIZE-1]
#   axis 3 → cities and misc  ∈ [0..N_PLAYERS+1]
#      z=0                 → tile description for city of player 0 
#      z=1                 → tile height for city of player 0  (0 if empty)
#      z=2                 → tile description for city of player 1
#      ...
#      z=2*N_PLAYERS-1     → tile height for city of player N_PLAYERS-1
#
#      z=2*N_PLAYERS       → per-player scalars packed. For position (q,r):
#         (0..N_PLAYERS-1            , 0..N_COLORS-1) → plazas[p, c], nb of valid plazas of color c for player p
#         (N_PLAYERS..2*N_PLAYERS-1  , 0..N_COLORS-1) → districts[p, c], nb of valid districts (weighted by height) of color c for player p
#         (2*N_PLAYERS..3*N_PLAYERS-1, 0)             → total_scores[p], current score of player p (ensure no overflow)
#         (2*N_PLAYERS..3*N_PLAYERS-1, 1)             → stones[p], current nb of stones for player p
#
#      z=2*N_PLAYERS+1     → global scalars packed. For position (q,r):
#         (0,0..CONSTR_SITE_SIZE-1)   → construction_site[i], tile id in i-th place of construction site (-1 for empty slots at the end)
#         (1,0..PACKED_TILES_BYTES-1) → tiles_bitpack, i-th bit is 1 if tile id i is available
#         (2,0..1)                    → misc: 0 = round number, 1 = remaining number of not visible stacks
#
# Tile description = color + (type * 8), see AkropolisConstants.py
# color       = description  % 8
# type        = description // 8
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

# @njit(cache=True, fastmath=True, nogil=True)
def observation_size():
	return (CITY_SIZE, CITY_SIZE, 2*N_PLAYERS+2)

# @njit(cache=True, fastmath=True, nogil=True)
def action_size():
	return CONSTR_SITE_SIZE * CITY_AREA * N_ORIENTS


#@njit(cache=True, fastmath=True, nogil=True)
def my_packbits(array):
	b = np.asarray(array, dtype=np.uint8)
	pad = (-b.size) % 8
	if pad:
		b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
	b = b.reshape(-1, 8)

	mask = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)

	packed = (b * mask).sum(axis=1).astype(np.int8)
	return packed

#@njit(cache=True, fastmath=True, nogil=True)
def my_unpackbits(values):
	p = np.asarray(values, dtype=np.int8).astype(np.uint8)

	mask = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)

	bits_matrix = (p[:, None] & mask) > 0
	bools = bits_matrix.reshape(-1)
	return bools

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

# spec = [
# 	('state'            , numba.int8[:,:,:]),
# 	('board'            , numba.int8[:,:,:,:]),
# 	('plazas'           , numba.int8[:,:]),
# 	('districts'        , numba.int8[:,:]),
# 	('total_scores'     , numba.int8[:]),
# 	('stones'           , numba.int8[:]),
# 	('construction_site', numba.int8[:]),
# 	('tiles_bitpack'    , numba.int8[:]),
# 	('misc'             , numba.int8[:]),
# ]
# We will enable such flag later
# @numba.experimental.jitclass(spec)
class Board():
	def __init__(self, num_players):
		self.state = np.zeros((CITY_SIZE, CITY_SIZE, 2*N_PLAYERS+2), dtype=np.int8)
		self.init_game()

	def init_game(self):
		self.copy_state(np.zeros((CITY_SIZE, CITY_SIZE, 2*N_PLAYERS+2), dtype=np.int8), copy_or_not=False)
		self.stones[:] = np.arange(1, N_PLAYERS+1, dtype=np.int8)
		self.tiles_bitpack[:] = my_packbits(TILES_DATA[:,3] <= N_PLAYERS)
		self.misc[1] = N_STACKS
		# Set initial tile
		self.board[START_TILE_R, START_TILE_Q, :, 0] = PLAZA*8 + BLUE
		self.board[START_TILE_R, START_TILE_Q, :, 1] = 1
		for idx in NEIGHBORS[START_TILE_R*CITY_SIZE+START_TILE_Q, ::2]:
			rr, qq = divmod(idx, CITY_SIZE)
			self.board[rr, qq, :, 0] = QUARRY*8
			self.board[rr, qq, :, 1] = 1

		# Build construction site
		self.construction_site[:] = -1
		self._draw_tiles_constr_site(initial_draw=True)
		# All other items are zero

	def copy_state(self, state, copy_or_not):
		if self.state is state and not copy_or_not:
			return
		self.state = state.copy() if copy_or_not else state

		self.board             = self.state[:, :, :2*N_PLAYERS].reshape(CITY_SIZE, CITY_SIZE, N_PLAYERS, 2)
		self.plazas            = self.state[           :  N_PLAYERS, :N_COLORS, 2*N_PLAYERS]
		self.districts         = self.state[  N_PLAYERS:2*N_PLAYERS, :N_COLORS, 2*N_PLAYERS]
		self.total_scores      = self.state[2*N_PLAYERS:3*N_PLAYERS, 0        , 2*N_PLAYERS]
		self.stones            = self.state[2*N_PLAYERS:3*N_PLAYERS, 1        , 2*N_PLAYERS]
		self.construction_site = self.state[0, :CONSTR_SITE_SIZE  , 2*N_PLAYERS+1]
		self.tiles_bitpack     = self.state[1, :PACKED_TILES_BYTES, 2*N_PLAYERS+1]
		# misc contains round number (0) and remaining number of stacks (1)
		self.misc              = self.state[2, :2                 , 2*N_PLAYERS+1]

	def make_move(self, move, player, random_seed):
		# decode "move" using divmod
		tile_idx_in_cs, rem  = divmod(move, N_PATTERNS)
		idx, orient          = divmod(rem, N_ORIENTS)

		# remove tile from construction_site, and move further tiles
		tile_id = self.construction_site[tile_idx_in_cs]
		self.construction_site[tile_idx_in_cs:-1] = self.construction_site[tile_idx_in_cs+1:]
		self.construction_site[-1] = -1
		for desc, hex_idx in zip(TILES_DATA[tile_id][:3], PATTERNS[idx*N_ORIENTS+orient]):
			rr, qq = divmod(hex_idx, CITY_SIZE)
			# update internals if building upon a quarry or plaza (district will be managed later)
			under_type, under_color = divmod(self.board[rr, qq, player, 0], 8)
			if under_type == PLAZA:
				self.plazas[player, under_color] += 1
			if under_type == QUARRY:
				self.stones[player] += 1
			self.board[rr, qq, player, 0] = desc
			self.board[rr, qq, player, 1] += 1
			# Update plazas
			if desc // 8 == PLAZA:
				self.plazas[player, desc % 8] += 1

		# update stones, districts, total_scores for current player
		self.stones[player] -= tile_idx_in_cs
		self._update_districts(player)
		total = (self.districts[player, :] * self.plazas[player, :]).sum() + self.stones[player]
		self.total_scores[player] = min(127, total)

		# Round number
		self.misc[0] += 1

		# if only 1 tile remaining in construction_site, draw a new set of tiles
		if self.construction_site[1] < 0 and self.misc[1] > 0:
			self._draw_tiles_constr_site(initial_draw=False)
			self.misc[1] -= 1 # remaining number of stacks

		return (player+1)%N_PLAYERS

	def valid_moves(self, player):
		heights_flat = self.board[:, :, player, 1].ravel()

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

			# Pattern is valid for placement
			pattern_is_valid[pattern_id] = True

		result = np.zeros(CONSTR_SITE_SIZE * N_PATTERNS, dtype=np.bool_)
		# For each available tile slot, apply the valid patterns
		for slot_index in range(CONSTR_SITE_SIZE):
			if self.construction_site[slot_index] < 0:
				# Skip empty slots
				continue

			base_offset = slot_index * N_PATTERNS
			for pattern_id in range(N_PATTERNS):
				if pattern_is_valid[pattern_id]:
					result[base_offset + pattern_id] = True

		return result

	def get_state(self):
		return self.state

	def check_end_game(self, next_player):
		if (self.misc[1] <= 0 and self.construction_site[1] < 0):
			m = self.total_scores.max()
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
		tmp_board = np.empty_like(self.board)
		tmp_plazas = np.empty_like(self.plazas)
		tmp_districts = np.empty_like(self.districts)
		tmp_scores = np.empty_like(self.total_scores)
		tmp_stones = np.empty_like(self.stones)

		# Roll data
		for p in range(N_PLAYERS):
			tmp_board[:, :, p, :] = self.board[:, :, (p + nb_swaps) % N_PLAYERS, :]
			tmp_plazas[p, :] = self.plazas[(p + nb_swaps) % N_PLAYERS, :]
			tmp_districts[p, :] = self.districts[(p + nb_swaps) % N_PLAYERS, :]
			tmp_scores[p] = self.total_scores[(p + nb_swaps) % N_PLAYERS]
			tmp_stones[p] = self.stones[(p + nb_swaps) % N_PLAYERS]

		self.board[:, :, :, :] = tmp_board
		self.plazas[:, :] = tmp_plazas
		self.districts[:, :] = tmp_districts
		self.total_scores[:] = tmp_scores
		self.stones[:] = tmp_stones

	def get_symmetries(self, policy, valids):
		# Always called on canonical board, meaning player = 0
		symmetries = [(self.state.copy(), policy.copy(), valids.copy())]
		state_backup, policy_backup, valids_backup = symmetries[0]

		# Rotate board by 60°, 120°, 180°, 240° and 300°

		# Translate board by +1 and -1 in q and r

		
		return symmetries

	def _draw_tiles_constr_site(self, initial_draw=False):
		available_tiles = my_unpackbits(self.tiles_bitpack)
		for i in range(0 if initial_draw else 1, CONSTR_SITE_SIZE):
			tile_id = np.random.choice(np.flatnonzero(available_tiles))
			self.construction_site[i] = tile_id
			available_tiles[tile_id] = False
		self.tiles_bitpack[:] = my_packbits(available_tiles)


	def _update_districts(self, player: int):
	    # 0) Récupère desc + hauteur et met à plat
	    desc2d   = self.board[:, :, player, 0]
	    h2d      = self.board[:, :, player, 1]
	    desc = desc2d.ravel()
	    h    = h2d.ravel()
	    district = np.zeros(5, dtype=np.int32)

	    # 1) GREEN (Gardens)
	    mask_green = desc == (DISTRICT*8 + GREEN)
	    district[GREEN] = h[mask_green].sum()

	    # # 2) YELLOW (Markets isolés)
	    # mask_yellow2d = (desc2d == (DISTRICT*8 + YELLOW)).astype(np.int32)
	    # nbr_y = np.zeros_like(mask_yellow2d)
	    # for di, dj in DIRECTIONS:
	    #     nbr_y += np.roll(np.roll(mask_yellow2d, di, axis=0), dj, axis=1)
	    # mask_yellow = desc == (DISTRICT*8 + YELLOW)
	    # district[YELLOW] = h[mask_yellow & (nbr_y.ravel() == 0)].sum()

	    # # 3) PURPLE (Temples entourés)
	    # occ2d = (h2d > 0).astype(np.int32)
	    # nbr_occ = np.zeros_like(occ2d)
	    # for di, dj in DIRECTIONS:
	    #     nbr_occ += np.roll(np.roll(occ2d, di, axis=0), dj, axis=1)
	    # mask_purple = desc == (DISTRICT*8 + PURPLE)
	    # district[PURPLE] = h[mask_purple & (nbr_occ.ravel() == 6)].sum()

	    # 4) Flood-fill des vides extérieurs
	    #    - empties reliés à bord du plateau
	    is_empty = desc == EMPTY
	    outer_empty = np.zeros_like(is_empty)
	    # init : empties qui ont un voisin <0 (bord)
	    for idx in np.nonzero(is_empty)[0]:
	        for nb in NEIGHBORS[idx]:
	            if nb < 0:
	                outer_empty[idx] = True
	                break
	    # BFS / DFS sur empty → mark all reachable
	    stack = [idx for idx in np.nonzero(outer_empty)[0]]
	    for cur in stack:
	        for nb in NEIGHBORS[cur]:
	            if nb < 0 or outer_empty[nb] or not is_empty[nb]:
	                continue
	            outer_empty[nb] = True
	            stack.append(nb)

	    # 5) RED (Barracks en périphérie réelle)
	    mask_red = desc == (DISTRICT*8 + RED)
	    # un RED compte s’il touche au moins un outer_empty
	    red_touch = np.zeros_like(mask_red)
	    for idx in np.nonzero(mask_red)[0]:
	        for nb in NEIGHBORS[idx]:
	            if nb < 0 or outer_empty[nb]:
	                red_touch[idx] = True
	                break
	    district[RED] = h[red_touch].sum()

	    # 6) BLUE (maisons) : BFS pour la plus longue chaîne
	    mask_blue = desc == (DISTRICT*8 + BLUE)
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
	        if chain > max_chain:
	            max_chain = chain
	    district[BLUE] = max_chain

	    # 7) Écriture des scores
	    self.districts[player, :] = district
