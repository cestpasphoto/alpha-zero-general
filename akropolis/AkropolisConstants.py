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
QUARRY   = 1
PLAZA    = 2
DISTRICT = 3

# Tile colors
BLUE   = 0
YELLOW = 1
RED    = 2
PURPLE = 3
GREEN  = 4
N_COLORS = 5

TYPECOL_LIST = [
	EMPTY,
	8*QUARRY,
	8*PLAZA+BLUE, 8*PLAZA+YELLOW, 8*PLAZA+RED, 8*PLAZA+PURPLE, 8*PLAZA+GREEN,
	8*DISTRICT+BLUE, 8*DISTRICT+YELLOW, 8*DISTRICT+RED, 8*DISTRICT+PURPLE, 8*DISTRICT+GREEN,
]

# Tile description = color + (type * 8)
# color       = description  % 8
# type        = description // 8

PLAZA_STARS = np.array([1, 2, 2, 2, 3], dtype=np.int8)

# Size of construction size (visible deck)
CONSTR_SITE_SIZE = N_PLAYERS + 1

# Deck size (note that construction size is not included)
N_STACKS = 11
DECK_SIZE = (N_PLAYERS+1) * N_STACKS

TOTAL_TILES = CONSTR_SITE_SIZE + DECK_SIZE # 2p = 36 (5B), 3p = 48 (6B), 4p = 60 (8B)

# =============================================================================

# odd-r offset
#                     SW        SE        E        NE        NW        W
DIRECTIONS_EVEN = [(-1, +1), ( 0, +1), (+1, 0), ( 0, -1), (-1, -1), (-1, 0)]
DIRECTIONS_ODD  = [( 0, +1), (+1, +1), (+1, 0), (+1, -1), ( 0, -1), (-1, 0)]

# Maximum number of orientations per tile
N_ORIENTS = 6

# Pattern = position of the 3 hexes of tile depending on
#   the position of its center (idx)
#   its orientation (orient)
# p = idx*N_ORIENTS+orient
N_PATTERNS = CITY_AREA * N_ORIENTS

# Size in bytes of total tiles stored in a bitfield
# At most 61 tiles, so fits into 8 bytes
PACKED_TILES_BYTES = 8

# =============================================================================

# Tiles description, and for which nb of players they are meant to 
# The order is such as the triangle points upwards.
# description = (color << 0) | (type << 3)
TILES_DATA = np.array([
	[ QUARRY*8           , PLAZA*8 + GREEN    , DISTRICT*8 + BLUE  , 2 ],  # Q G* B  ; 2pl
	[ QUARRY*8           , PLAZA*8 + GREEN    , QUARRY*8           , 2 ],  # Q G* Q  ; 2pl
	[ DISTRICT*8 + BLUE  , PLAZA*8 + GREEN    , QUARRY*8           , 2 ],  # B G* Q  ; 2pl
	[ DISTRICT*8 + RED   , PLAZA*8 + PURPLE   , DISTRICT*8 + BLUE  , 2 ],  # R P* B  ; 2pl
	[ DISTRICT*8 + BLUE  , PLAZA*8 + PURPLE   , QUARRY*8           , 2 ],  # B P* Q  ; 2pl
	[ QUARRY*8           , PLAZA*8 + PURPLE   , QUARRY*8           , 2 ],  # Q P* Q  ; 2pl
	[ QUARRY*8           , PLAZA*8 + PURPLE   , DISTRICT*8 + BLUE  , 2 ],  # Q P* B  ; 2pl
	[ DISTRICT*8 + PURPLE, PLAZA*8 + RED      , DISTRICT*8 + BLUE  , 2 ],  # P R* B  ; 2pl
	[ QUARRY*8           , PLAZA*8 + RED      , QUARRY*8           , 2 ],  # Q R* Q  ; 2pl
	[ QUARRY*8           , PLAZA*8 + RED      , DISTRICT*8 + BLUE  , 2 ],  # Q R* B  ; 2pl
	[ DISTRICT*8 + BLUE  , PLAZA*8 + RED      , DISTRICT*8 + YELLOW, 2 ],  # B R* Y  ; 2pl
	[ QUARRY*8           , PLAZA*8 + YELLOW   , QUARRY*8           , 2 ],  # Q Y* Q  ; 2pl
	[ DISTRICT*8 + BLUE  , PLAZA*8 + YELLOW   , DISTRICT*8 + PURPLE, 2 ],  # B Y* P  ; 2pl
	[ QUARRY*8           , PLAZA*8 + YELLOW   , DISTRICT*8 + RED   , 2 ],  # Q Y* R  ; 2pl
	[ DISTRICT*8 + GREEN , PLAZA*8 + YELLOW   , DISTRICT*8 + BLUE  , 2 ],  # G Y* B  ; 2pl
	[ QUARRY*8           , PLAZA*8 + BLUE     , QUARRY*8           , 2 ],  # Q B* Q  ; 2pl
	[ QUARRY*8           , PLAZA*8 + BLUE     , QUARRY*8           , 2 ],  # Q B* Q  ; 2pl
	[ QUARRY*8           , PLAZA*8 + BLUE     , DISTRICT*8 + GREEN , 2 ],  # Q B* G  ; 2pl
	[ DISTRICT*8 + RED   , PLAZA*8 + BLUE     , DISTRICT*8 + YELLOW, 2 ],  # R B* Y  ; 2pl
	[ QUARRY*8           , PLAZA*8 + BLUE     , QUARRY*8           , 2 ],  # Q B* Q  ; 2pl
	[ QUARRY*8           , DISTRICT*8 + PURPLE, QUARRY*8           , 2 ],  # Q P Q   ; 2pl
	[ DISTRICT*8 + PURPLE, QUARRY*8           , DISTRICT*8 + YELLOW, 2 ],  # P Q Y   ; 2pl
	[ DISTRICT*8 + BLUE  , QUARRY*8           , DISTRICT*8 + BLUE  , 2 ],  # B Q B   ; 2pl
	[ QUARRY*8           , DISTRICT*8 + YELLOW, QUARRY*8           , 2 ],  # Q Y Q   ; 2pl
	[ DISTRICT*8 + YELLOW, QUARRY*8           , DISTRICT*8 + RED   , 2 ],  # Y Q R   ; 2pl
	[ DISTRICT*8 + RED   , QUARRY*8           , DISTRICT*8 + BLUE  , 2 ],  # R Q B   ; 2pl
	[ DISTRICT*8 + BLUE  , QUARRY*8           , DISTRICT*8 + YELLOW, 2 ],  # B Q Y   ; 2pl
	[ DISTRICT*8 + YELLOW, DISTRICT*8 + BLUE  , DISTRICT*8 + PURPLE, 2 ],  # Y B P   ; 2pl
	[ QUARRY*8           , DISTRICT*8 + GREEN , QUARRY*8           , 2 ],  # Q G Q   ; 2pl
	[ DISTRICT*8 + YELLOW, DISTRICT*8 + BLUE  , DISTRICT*8 + GREEN , 2 ],  # Y B G   ; 2pl
	[ DISTRICT*8 + RED   , QUARRY*8           , DISTRICT*8 + PURPLE, 2 ],  # R Q P   ; 2pl
	[ DISTRICT*8 + YELLOW, QUARRY*8           , DISTRICT*8 + PURPLE, 2 ],  # Y Q P   ; 2pl
	[ DISTRICT*8 + RED   , DISTRICT*8 + BLUE  , DISTRICT*8 + GREEN , 2 ],  # R B G   ; 2pl
	[ DISTRICT*8 + RED   , QUARRY*8           , DISTRICT*8 + YELLOW, 2 ],  # R Q Y   ; 2pl
	[ DISTRICT*8 + PURPLE, DISTRICT*8 + BLUE  , DISTRICT*8 + RED   , 2 ],  # P B R   ; 2pl
	[ DISTRICT*8 + GREEN , QUARRY*8           , DISTRICT*8 + YELLOW, 2 ],  # G Q Y   ; 2pl
	[ DISTRICT*8 + YELLOW, QUARRY*8           , DISTRICT*8 + RED   , 2 ],  # Y Q R   ; 2pl

	[ DISTRICT*8 + RED   , PLAZA*8 + GREEN    , DISTRICT*8 + BLUE  , 3 ],  # R G* B  ; 3pl
	[ QUARRY*8           , PLAZA*8 + PURPLE   , QUARRY*8           , 3 ],  # Q P* Q  ; 3pl
	[ DISTRICT*8 + BLUE  , PLAZA*8 + RED      , QUARRY*8           , 3 ],  # B R* Q  ; 3pl
	[ QUARRY*8           , PLAZA*8 + YELLOW   , QUARRY*8           , 3 ],  # Q Y* Q  ; 3pl
	[ DISTRICT*8 + YELLOW, PLAZA*8 + BLUE     , DISTRICT*8 + PURPLE, 3 ],  # Y B* P  ; 3pl
	[ QUARRY*8           , DISTRICT*8 + BLUE  , QUARRY*8           , 3 ],  # Q B Q   ; 3pl
	[ DISTRICT*8 + GREEN , QUARRY*8           , DISTRICT*8 + RED   , 3 ],  # G Q R   ; 3pl
	[ DISTRICT*8 + BLUE  , QUARRY*8           , DISTRICT*8 + YELLOW, 3 ],  # B Q Y   ; 3pl
	[ DISTRICT*8 + BLUE  , QUARRY*8           , DISTRICT*8 + PURPLE, 3 ],  # B Q P   ; 3pl
	[ DISTRICT*8 + YELLOW, QUARRY*8           , DISTRICT*8 + BLUE  , 3 ],  # Y Q B   ; 3pl
	[ DISTRICT*8 + BLUE  , QUARRY*8           , DISTRICT*8 + BLUE  , 3 ],  # B Q B   ; 3pl
	[ DISTRICT*8 + RED   , DISTRICT*8 + BLUE  , DISTRICT*8 + YELLOW, 3 ],  # R B Y   ; 3pl

	[ DISTRICT*8 + BLUE  , PLAZA*8 + GREEN    , DISTRICT*8 + YELLOW, 4 ],  # B G* Y  ; 4pl
	[ DISTRICT*8 + YELLOW, PLAZA*8 + PURPLE   , DISTRICT*8 + BLUE  , 4 ],  # Y P* B  ; 4pl
	[ QUARRY*8           , PLAZA*8 + RED      , QUARRY*8           , 4 ],  # Q R* Q  ; 4pl
	[ DISTRICT*8 + PURPLE, PLAZA*8 + YELLOW   , QUARRY*8           , 4 ],  # P Y* Q  ; 4pl
	[ DISTRICT*8 + YELLOW, PLAZA*8 + BLUE     , QUARRY*8           , 4 ],  # Y B* Q  ; 4pl
	[ DISTRICT*8 + BLUE  , QUARRY*8           , DISTRICT*8 + RED   , 4 ],  # B Q R   ; 4pl
	[ DISTRICT*8 + PURPLE, QUARRY*8           , DISTRICT*8 + BLUE  , 4 ],  # P Q B   ; 4pl
	[ DISTRICT*8 + BLUE  , QUARRY*8           , DISTRICT*8 + GREEN , 4 ],  # B Q G   ; 4pl
	[ QUARRY*8           , DISTRICT*8 + RED   , QUARRY*8           , 4 ],  # Q R Q   ; 4pl
	[ DISTRICT*8 + RED   , QUARRY*8           , DISTRICT*8 + BLUE  , 4 ],  # R Q B   ; 4pl
	[ DISTRICT*8 + YELLOW, QUARRY*8           , DISTRICT*8 + BLUE  , 4 ],  # Y Q B   ; 4pl
	[ DISTRICT*8 + BLUE  , QUARRY*8           , DISTRICT*8 + BLUE  , 4 ],  # B Q B   ; 4pl
], dtype=np.int8)

# =============================================================================

#