import numpy as np

N_PLAYERS = 2

# Size of construction size (visible deck)
CONSTR_SITE_SIZE = N_PLAYERS + 1

# Deck size (note that construction size is not included)
N_STACKS = 11

# Tile descriptions
EMPTY          = 0
QUARRY         = 1
DISTRICT_BLUE  = 2
DISTRICT_YELLOW= 3
DISTRICT_RED   = 4
DISTRICT_PURPLE= 5
DISTRICT_GREEN = 6
PLAZA_BLUE     = 7
PLAZA_YELLOW   = 8
PLAZA_RED      = 9
PLAZA_PURPLE   =10
PLAZA_GREEN    =11
CODES_LIST = np.array([
	EMPTY, QUARRY, DISTRICT_BLUE, DISTRICT_YELLOW, DISTRICT_RED, DISTRICT_PURPLE, DISTRICT_GREEN, PLAZA_BLUE, PLAZA_YELLOW, PLAZA_RED, PLAZA_PURPLE, PLAZA_GREEN,
], dtype=np.int8)	
# Color
BLUE   = 0
YELLOW = 1
RED    = 2
PURPLE = 3
GREEN  = 4
N_COLORS = 5
# Category
EMPTY    = 0
QUARRY   = 1
DISTRICT = 2
PLAZA    = 3
# Conversion tables
DESCR_TO_TYPE_COLOR = np.array([
	[EMPTY,    0],      #  0 = EMPTY
	[QUARRY,   0],      #  1 = QUARRY
	[DISTRICT, BLUE],   #  2 = DISTRICT_BLUE
	[DISTRICT, YELLOW], #  3 = DISTRICT_YELLOW
	[DISTRICT, RED],    #  4 = DISTRICT_RED
	[DISTRICT, PURPLE], #  5 = DISTRICT_PURPLE
	[DISTRICT, GREEN],  #  6 = DISTRICT_GREEN
	[PLAZA,    BLUE],   #  7 = PLAZA_BLUE
	[PLAZA,    YELLOW], #  8 = PLAZA_YELLOW
	[PLAZA,    RED],    #  9 = PLAZA_RED
	[PLAZA,    PURPLE], # 10 = PLAZA_PURPLE
	[PLAZA,    GREEN],  # 11 = PLAZA_GREEN
], dtype=np.int8)
TYPE_COLOR_TO_DESCR = np.array([
	[0, -1, -1, -1, -1],  # type=0 empty
	[1, -1, -1, -1, -1],  # type=1 quarry
	[2,  3,  4,  5,  6],  # type=2 district: codes 2…6
	[7,  8,  9, 10, 11],  # type=3 plaza   : codes 7…11
], dtype=np.int8)

# Nb stars per plaza color
PLAZA_STARS = np.array([1, 2, 2, 2, 3], dtype=np.int8)


# =============================================================================
# Implementation specific

# City 1D size
CITY_SIZE = 12
# Total number of board positions
CITY_AREA = CITY_SIZE * CITY_SIZE
# Position of the initial tile position
START_TILE_Q, START_TILE_R = (2*CITY_SIZE)//5, (2*CITY_SIZE)//5

# Maximum number of orientations per tile
N_ORIENTS = 6

# odd-r offset
#                     SW        SE        E        NE        NW        W
DIRECTIONS_EVEN = [(-1, +1), ( 0, +1), (+1, 0), ( 0, -1), (-1, -1), (-1, 0)]
DIRECTIONS_ODD  = [( 0, +1), (+1, +1), (+1, 0), (+1, -1), ( 0, -1), (-1, 0)]

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
TILES_DATA = np.array([
	[ QUARRY             , PLAZA_GREEN        , DISTRICT_BLUE      , 2 ],  # Q G* B  ; 2pl
	[ QUARRY             , PLAZA_GREEN        , QUARRY             , 2 ],  # Q G* Q  ; 2pl
	[ DISTRICT_BLUE      , PLAZA_GREEN        , QUARRY             , 2 ],  # B G* Q  ; 2pl
	[ DISTRICT_RED       , PLAZA_PURPLE       , DISTRICT_BLUE      , 2 ],  # R P* B  ; 2pl
	[ DISTRICT_BLUE      , PLAZA_PURPLE       , QUARRY             , 2 ],  # B P* Q  ; 2pl
	[ QUARRY             , PLAZA_PURPLE       , QUARRY             , 2 ],  # Q P* Q  ; 2pl
	[ QUARRY             , PLAZA_PURPLE       , DISTRICT_BLUE      , 2 ],  # Q P* B  ; 2pl
	[ DISTRICT_PURPLE    , PLAZA_RED          , DISTRICT_BLUE      , 2 ],  # P R* B  ; 2pl
	[ QUARRY             , PLAZA_RED          , QUARRY             , 2 ],  # Q R* Q  ; 2pl
	[ QUARRY             , PLAZA_RED          , DISTRICT_BLUE      , 2 ],  # Q R* B  ; 2pl
	[ DISTRICT_BLUE      , PLAZA_RED          , DISTRICT_YELLOW    , 2 ],  # B R* Y  ; 2pl
	[ QUARRY             , PLAZA_YELLOW       , QUARRY             , 2 ],  # Q Y* Q  ; 2pl
	[ DISTRICT_BLUE      , PLAZA_YELLOW       , DISTRICT_PURPLE    , 2 ],  # B Y* P  ; 2pl
	[ QUARRY             , PLAZA_YELLOW       , DISTRICT_RED       , 2 ],  # Q Y* R  ; 2pl
	[ DISTRICT_GREEN     , PLAZA_YELLOW       , DISTRICT_BLUE      , 2 ],  # G Y* B  ; 2pl
	[ QUARRY             , PLAZA_BLUE         , QUARRY             , 2 ],  # Q B* Q  ; 2pl
	[ QUARRY             , PLAZA_BLUE         , QUARRY             , 2 ],  # Q B* Q  ; 2pl
	[ QUARRY             , PLAZA_BLUE         , DISTRICT_GREEN     , 2 ],  # Q B* G  ; 2pl
	[ DISTRICT_RED       , PLAZA_BLUE         , DISTRICT_YELLOW    , 2 ],  # R B* Y  ; 2pl
	[ QUARRY             , PLAZA_BLUE         , QUARRY             , 2 ],  # Q B* Q  ; 2pl
	[ QUARRY             , DISTRICT_PURPLE    , QUARRY             , 2 ],  # Q P Q   ; 2pl
	[ DISTRICT_PURPLE    , QUARRY             , DISTRICT_YELLOW    , 2 ],  # P Q Y   ; 2pl
	[ DISTRICT_BLUE      , QUARRY             , DISTRICT_BLUE      , 2 ],  # B Q B   ; 2pl
	[ QUARRY             , DISTRICT_YELLOW    , QUARRY             , 2 ],  # Q Y Q   ; 2pl
	[ DISTRICT_YELLOW    , QUARRY             , DISTRICT_RED       , 2 ],  # Y Q R   ; 2pl
	[ DISTRICT_RED       , QUARRY             , DISTRICT_BLUE      , 2 ],  # R Q B   ; 2pl
	[ DISTRICT_BLUE      , QUARRY             , DISTRICT_YELLOW    , 2 ],  # B Q Y   ; 2pl
	[ DISTRICT_YELLOW    , DISTRICT_BLUE      , DISTRICT_PURPLE    , 2 ],  # Y B P   ; 2pl
	[ QUARRY             , DISTRICT_GREEN     , QUARRY             , 2 ],  # Q G Q   ; 2pl
	[ DISTRICT_YELLOW    , DISTRICT_BLUE      , DISTRICT_GREEN     , 2 ],  # Y B G   ; 2pl
	[ DISTRICT_RED       , QUARRY             , DISTRICT_PURPLE    , 2 ],  # R Q P   ; 2pl
	[ DISTRICT_YELLOW    , QUARRY             , DISTRICT_PURPLE    , 2 ],  # Y Q P   ; 2pl
	[ DISTRICT_RED       , DISTRICT_BLUE      , DISTRICT_GREEN     , 2 ],  # R B G   ; 2pl
	[ DISTRICT_RED       , QUARRY             , DISTRICT_YELLOW    , 2 ],  # R Q Y   ; 2pl
	[ DISTRICT_PURPLE    , DISTRICT_BLUE      , DISTRICT_RED       , 2 ],  # P B R   ; 2pl
	[ DISTRICT_GREEN     , QUARRY             , DISTRICT_YELLOW    , 2 ],  # G Q Y   ; 2pl
	[ DISTRICT_YELLOW    , QUARRY             , DISTRICT_RED       , 2 ],  # Y Q R   ; 2pl

	[ DISTRICT_RED       , PLAZA_GREEN        , DISTRICT_BLUE      , 3 ],  # R G* B  ; 3pl
	[ QUARRY             , PLAZA_PURPLE       , QUARRY             , 3 ],  # Q P* Q  ; 3pl
	[ DISTRICT_BLUE      , PLAZA_RED          , QUARRY             , 3 ],  # B R* Q  ; 3pl
	[ QUARRY             , PLAZA_YELLOW       , QUARRY             , 3 ],  # Q Y* Q  ; 3pl
	[ DISTRICT_YELLOW    , PLAZA_BLUE         , DISTRICT_PURPLE    , 3 ],  # Y B* P  ; 3pl
	[ QUARRY             , DISTRICT_BLUE      , QUARRY             , 3 ],  # Q B Q   ; 3pl
	[ DISTRICT_GREEN     , QUARRY             , DISTRICT_RED       , 3 ],  # G Q R   ; 3pl
	[ DISTRICT_BLUE      , QUARRY             , DISTRICT_YELLOW    , 3 ],  # B Q Y   ; 3pl
	[ DISTRICT_BLUE      , QUARRY             , DISTRICT_PURPLE    , 3 ],  # B Q P   ; 3pl
	[ DISTRICT_YELLOW    , QUARRY             , DISTRICT_BLUE      , 3 ],  # Y Q B   ; 3pl
	[ DISTRICT_BLUE      , QUARRY             , DISTRICT_BLUE      , 3 ],  # B Q B   ; 3pl
	[ DISTRICT_RED       , DISTRICT_BLUE      , DISTRICT_YELLOW    , 3 ],  # R B Y   ; 3pl

	[ DISTRICT_BLUE      , PLAZA_GREEN        , DISTRICT_YELLOW    , 4 ],  # B G* Y  ; 4pl
	[ DISTRICT_YELLOW    , PLAZA_PURPLE       , DISTRICT_BLUE      , 4 ],  # Y P* B  ; 4pl
	[ QUARRY             , PLAZA_RED          , QUARRY             , 4 ],  # Q R* Q  ; 4pl
	[ DISTRICT_PURPLE    , PLAZA_YELLOW       , QUARRY             , 4 ],  # P Y* Q  ; 4pl
	[ DISTRICT_YELLOW    , PLAZA_BLUE         , QUARRY             , 4 ],  # Y B* Q  ; 4pl
	[ DISTRICT_BLUE      , QUARRY             , DISTRICT_RED       , 4 ],  # B Q R   ; 4pl
	[ DISTRICT_PURPLE    , QUARRY             , DISTRICT_BLUE      , 4 ],  # P Q B   ; 4pl
	[ DISTRICT_BLUE      , QUARRY             , DISTRICT_GREEN     , 4 ],  # B Q G   ; 4pl
	[ QUARRY             , DISTRICT_RED       , QUARRY             , 4 ],  # Q R Q   ; 4pl
	[ DISTRICT_RED       , QUARRY             , DISTRICT_BLUE      , 4 ],  # R Q B   ; 4pl
	[ DISTRICT_YELLOW    , QUARRY             , DISTRICT_BLUE      , 4 ],  # Y Q B   ; 4pl
	[ DISTRICT_BLUE      , QUARRY             , DISTRICT_BLUE      , 4 ],  # B Q B   ; 4pl
], dtype=np.int8)

# =============================================================================

#