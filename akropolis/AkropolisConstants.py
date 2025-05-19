import numpy as np

N_PLAYERS = 2

# City 1D size
CITY_SIZE = 12
# Total number of board positions
CITY_AREA = CITY_SIZE * CITY_SIZE
# Position of the initial tile position
START_TILE_Q, START_TILE_R = (2*CITY_SIZE)//5, (2*CITY_SIZE)//5

# Tile types
EMPTY    = 0
QUARRY   = 2
PLAZA    = 3
DISTRICT = 4

# Tile colors
BLUE   = 0
YELLOW = 1
RED    = 2
PURPLE = 3
GREEN  = 4
N_COLORS = 5

# Tile description = color + (type * 8)
# color       = description  % 8
# type        = description // 8

# Size of construction size (visible deck)
CONSTR_SITE_SIZE = N_PLAYERS + 1

# Deck size (note that construction size is not included)
N_STACKS = 11
DECK_SIZE = (N_PLAYERS+1) * N_STACKS

TOTAL_TILES = CONSTR_SITE_SIZE + DECK_SIZE # 2p = 36 (5B), 3p = 48 (6B), 4p = 60 (8B)

# =============================================================================

# SW, SE, E, NE, NW, W
DIRECTIONS = [(-1,1), (0,1), (1,0), (1,-1), (0,-1), (-1,0)]

# Maximum number of orientations per tile
N_ORIENTS = 6

# Size in bytes of total tiles stored in a bitfield
# At most 61 tiles, so fits into 8 bytes
PACKED_TILES_BYTES = 8

# Pre-computed neighbors
NEIGHBORS = np.full((CITY_AREA, 6), -1, dtype=np.int16)
for i in range(CITY_SIZE):
	for j in range(CITY_SIZE):
		idx = i * CITY_SIZE + j
		cnt = 0
		for di, dj in DIRECTIONS:
			ni, nj = i + di, j + dj
			if 0 <= ni < CITY_SIZE and 0 <= nj < CITY_SIZE:
				NEIGHBORS[idx, cnt] = ni * CITY_SIZE + nj
				cnt += 1

# PATTERNS[p] = (n1, s, n2) or (-1, -1, -1)
N_PATTERNS = CITY_AREA * N_ORIENTS
PATTERNS = np.full((N_PATTERNS, 3), -1, dtype=np.int16)
for s in range(CITY_AREA):
	q, r = divmod(s, CITY_SIZE)
	for o in range(N_ORIENTS):
		idx = s * N_ORIENTS + o
		d1 = DIRECTIONS[o]
		d2 = DIRECTIONS[(o + 1) % N_ORIENTS]
		pts = [
			(q + d1[0], r + d1[1]),
			(q        , r),
			(q + d2[0], r + d2[1]),
		]
		if all(0 <= qq < CITY_SIZE and 0 <= rr < CITY_SIZE for qq, rr in pts):
			for j, (qq, rr) in enumerate(pts):
				PATTERNS[idx, j] = qq * CITY_SIZE + rr

# PATTERN_NEI[p] = list of 9 neighbours around the tile
PATTERN_NEI = np.full((N_PATTERNS, 9), -1, dtype=np.int16)
for p in range(N_PATTERNS):
	triplet = PATTERNS[p]
	nbrs = set()
	for cell in triplet:
		if cell < 0:
			continue
		for neighbor in NEIGHBORS[cell]:
			if neighbor not in triplet:
				nbrs.add(neighbor)
	PATTERN_NEI[p, :len(nbrs)] = sorted(nbrs)

# =============================================================================

# Tiles description, and for which nb of players they are meant to 
# The order is such as the triangle points upwards.
# description = (color << 0) | (type << 3)
TILES_DATA = np.array([
	[ 16, 28, 32, 2 ],  # Q G* B  ; 2pl
	[ 16, 28, 16, 2 ],  # Q G* Q  ; 2pl
	[ 32, 28, 16, 2 ],  # B G* Q  ; 2pl
	[ 34, 27, 32, 2 ],  # R P* B  ; 2pl
	[ 32, 27, 16, 2 ],  # B P* Q  ; 2pl
	[ 16, 27, 16, 2 ],  # Q P* Q  ; 2pl
	[ 16, 27, 32, 2 ],  # Q P* B  ; 2pl
	[ 35, 26, 32, 2 ],  # P R* B  ; 2pl
	[ 16, 26, 16, 2 ],  # Q R* Q  ; 2pl
	[ 16, 26, 32, 2 ],  # Q R* B  ; 2pl
	[ 32, 26, 33, 2 ],  # B R* Y  ; 2pl
	[ 16, 25, 16, 2 ],  # Q Y* Q  ; 2pl
	[ 32, 25, 35, 2 ],  # B Y* P  ; 2pl
	[ 16, 25, 34, 2 ],  # Q Y* R  ; 2pl
	[ 36, 25, 32, 2 ],  # G Y* B  ; 2pl
	[ 16, 24, 16, 2 ],  # Q B* Q  ; 2pl
	[ 16, 24, 16, 2 ],  # Q B* Q  ; 2pl
	[ 16, 24, 36, 2 ],  # Q B* G  ; 2pl
	[ 34, 24, 33, 2 ],  # R B* Y  ; 2pl
	[ 16, 24, 16, 2 ],  # Q B* Q  ; 2pl
	[ 16, 35, 16, 2 ],  # Q P Q   ; 2pl
	[ 35, 16, 33, 2 ],  # P Q Y   ; 2pl
	[ 32, 16, 32, 2 ],  # B Q B   ; 2pl
	[ 16, 33, 16, 2 ],  # Q Y Q   ; 2pl
	[ 33, 16, 34, 2 ],  # Y Q R   ; 2pl
	[ 34, 16, 32, 2 ],  # R Q B   ; 2pl
	[ 32, 16, 33, 2 ],  # B Q Y   ; 2pl
	[ 33, 32, 35, 2 ],  # Y B P   ; 2pl
	[ 16, 36, 16, 2 ],  # Q G Q   ; 2pl
	[ 33, 32, 36, 2 ],  # Y B G   ; 2pl
	[ 34, 16, 35, 2 ],  # R Q P   ; 2pl
	[ 33, 16, 35, 2 ],  # Y Q P   ; 2pl
	[ 34, 32, 36, 2 ],  # R B G   ; 2pl
	[ 34, 16, 33, 2 ],  # R Q Y   ; 2pl
	[ 35, 32, 34, 2 ],  # P B R   ; 2pl
	[ 36, 16, 33, 2 ],  # G Q Y   ; 2pl
	[ 33, 16, 34, 2 ],  # Y Q R   ; 2pl

	[ 34, 28, 32, 3 ],  # R G* B  ; 3pl
	[ 16, 27, 16, 3 ],  # Q P* Q  ; 3pl
	[ 32, 26, 16, 3 ],  # B R* Q  ; 3pl
	[ 16, 25, 16, 3 ],  # Q Y* Q  ; 3pl
	[ 33, 24, 35, 3 ],  # Y B* P  ; 3pl
	[ 16, 32, 16, 3 ],  # Q B Q   ; 3pl
	[ 36, 16, 34, 3 ],  # G Q R   ; 3pl
	[ 32, 16, 33, 3 ],  # B Q Y   ; 3pl
	[ 32, 16, 35, 3 ],  # B Q P   ; 3pl
	[ 33, 16, 32, 3 ],  # Y Q B   ; 3pl
	[ 32, 16, 32, 3 ],  # B Q B   ; 3pl
	[ 34, 32, 33, 3 ],  # R B Y   ; 3pl

	[ 32, 28, 33, 4 ],  # B G* Y  ; 4pl
	[ 33, 27, 32, 4 ],  # Y P* B  ; 4pl
	[ 16, 26, 16, 4 ],  # Q R* Q  ; 4pl
	[ 35, 25, 16, 4 ],  # P Y* Q  ; 4pl
	[ 33, 24, 16, 4 ],  # Y B* Q  ; 4pl
	[ 32, 16, 34, 4 ],  # B Q R   ; 4pl
	[ 35, 16, 32, 4 ],  # P Q B   ; 4pl
	[ 32, 16, 36, 4 ],  # B Q G   ; 4pl
	[ 16, 34, 16, 4 ],  # Q R Q   ; 4pl
	[ 34, 16, 32, 4 ],  # R Q B   ; 4pl
	[ 33, 16, 32, 4 ],  # Y Q B   ; 4pl
	[ 32, 16, 32, 4 ],  # B Q B   ; 4pl
], dtype=np.int8)

# =============================================================================

#