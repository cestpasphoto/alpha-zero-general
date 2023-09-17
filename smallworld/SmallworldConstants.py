import numpy as np
# from numba import njit

NUMBER_PLAYERS = 2
MAX_REDEPLOY = 5
DECK_SIZE = 6

DICE_VALUES = [0, 0, 0, 1, 2, 3]
AVG_DICE = 1
MAX_DICE = 3

# CONSTANTS
WATER    = 0
SAND     = 1
LAND     = 2
MOUNTAIN = 3
MEADOW   = 4

NOPOWR = 0
SOURCE = 1
MINE   = 2

NOPPL    = 0
PRIMIT   = 1
OGRE     = 2
DWARF    = 3
HUMAN    = 4
GIANT    = 5
AMAZON   = 6
SORCER   = 7
RATMAN   = 8
SKELETON = 9
TROLL    = 10 # update _init_deck()


NEW_TURN_STARTED  = 1
JUST_ATTACKED     = 2
JUST_ABANDONED    = 3
JUST_DECLINED     = 4
TO_START_REDEPLOY = 5
TO_REDEPLOY       = 6
WAITING_OTHER_PL  = 7

#################### MAP DESCRIPTION ####################

# Limits description
connexity_matrix = [
# 0  1  2  3  4  5  6  7  8 
[ 0, 1, 1, 0, 0, 0, 1, 0, 0,], # 0
[ 1, 0, 1, 0, 1, 1, 1, 1, 0,], # 1
[ 1, 1, 0, 1, 1, 0, 0, 0, 0,], # 2
[ 0, 0, 1, 0, 1, 0, 0, 0, 1,], # 3
[ 0, 1, 1, 1, 0, 1, 0, 0, 1,], # 4
[ 0, 1, 0, 0, 1, 0, 0, 1, 1,], # 5
[ 1, 1, 0, 0, 0, 0, 0, 1, 0,], # 6
[ 0, 1, 0, 0, 0, 1, 1, 0, 0,], # 7
[ 0, 0, 0, 1, 1, 1, 0, 0, 0,], # 8
]

assert(not np.any(np.transpose(connexity_matrix) - connexity_matrix))

NB_AREAS = 9

# Area description (terrain)
#  Terrain    power   edge   primitive 
descr = [
 [ WATER    , NOPOWR, True  ,  False], #0
 [ SAND     , MINE  , False ,  False], #1
 [ LAND     , SOURCE, True  ,  True ], #2
 [ WATER    , NOPOWR, True  ,  False], #3
 [ MOUNTAIN , NOPOWR, False ,  False], #4
 [ LAND     , SOURCE, True  ,  True ], #5
 [ MEADOW   , NOPOWR, True  ,  False], #6
 [ MOUNTAIN , MINE  , True  ,  True ], #7
 [ SAND     , MINE  , True  ,  True ], #8
]

#################### SPECIFIC TO DISPLAY ####################

DISPLAY_WIDTH, DISPLAY_HEIGHT = 7, 10
map_display = [
 [0,0,0,0,6,6,7,],
 [0,0,0,6,6,7,7,],
 [0,1,1,1,6,7,7,],
 [2,1,1,1,1,1,7,],
 [2,1,1,1,5,5,5,],
 [2,2,2,4,5,5,5,],
 [2,2,2,4,4,8,8,],
 [2,2,2,4,4,8,8,],
 [3,3,3,3,3,8,8,],
 [3,3,3,3,3,3,8,],
]

txt_display = [   # nb ppl (1), area ID (2), power (3)
 [0,2,0,0,2,0,0,],
 [1,3,0,1,3,0,2,],
 [0,0,0,0,0,1,3,],
 [0,0,1,3,0,0,0,],
 [0,0,2,0,2,0,3,],
 [0,0,0,0,0,0,1,],
 [0,1,3,0,2,0,0,],
 [0,2,0,1,3,0,2,],
 [0,0,1,3,0,1,3,],
 [0,0,0,2,0,0,0,],
]