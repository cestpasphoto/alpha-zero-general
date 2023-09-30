import numpy as np
# from numba import njit

NUMBER_PLAYERS = 2
MAX_REDEPLOY = 8
DECK_SIZE = 6

DICE_VALUES = [0, 0, 0, 1, 2, 3]
AVG_DICE = 1
MAX_DICE = 3

IMMUNE_CONQUEST = 40
FULL_IMMUNITY   = 120

# CONSTANTS
WATER    = 0
SAND     = 1
FARMLAND = 2
MOUNTAIN = 3
MEADOW   = 4

NOPOWR = 0
SOURCE = 1
MINE   = 2
MAGIC  = 3

NOPPL    = 0
AMAZON   = 1  #  +4 pour attaque                                         DONE
DWARF    = 2  #  +1 victoire sur mine, même en déclin                    DONE
ELF      = 3  #  pas de défausse lors d'une défaite                      DONE
GHOUL    = 4  #  tous les zombies restent en déclin, peuvent attaquer    DONE
GIANT    = 5  #  -1 pour attaque voisin montagne                         DONE
HALFLING = 6  #  départ n'importe où, immunité sur 2 prem régions        DONE
HUMAN    = 7  #  +1 victoire sur champs                                  DONE
ORC      = 8  #  +1 victoire pour région non-vide conquise               DONE
RATMAN   = 9  #  leur nombre                                             
SKELETON = 10 #  +1 pion pour toutes 2 régions non-vide conquises        DONE
SORCERER = 11 #  remplace pion unique adversaire actif par un sorcier    L
TRITON   = 12 #  -1 pour attaque région côtière                          DONE
TROLL    = 13 #  +1 défense sur chaque territoire même en déclin         DONE
WIZARD   = 14 #  +1 victoire sur source magique                          DONE
PRIMIT   = 15

MAX_SKELETONS = 20
MAX_SORCERERS = 18
#                       1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
initial_nb_people = [0, 6, 3, 6, 5, 6, 6, 5, 5, 8, 6, 5, 6, 5, 5, 2]
initial_tokens    = [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]

NEW_TURN_STARTED  = 1
JUST_ATTACKED     = 2
JUST_ABANDONED    = 3
JUST_DECLINED     = 4
NEED_ABANDON      = 5
TO_START_REDEPLOY = 6
TO_REDEPLOY       = 7
WAITING_OTHER_PL  = 8

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
 [ FARMLAND , SOURCE, True  ,  True ], #2
 [ WATER    , NOPOWR, True  ,  False], #3
 [ MOUNTAIN , NOPOWR, False ,  False], #4
 [ FARMLAND , SOURCE, True  ,  True ], #5
 [ MEADOW   , NOPOWR, True  ,  False], #6
 [ MOUNTAIN , MINE  , True  ,  True ], #7
 [ SAND     , MAGIC , True  ,  True ], #8
]

#################### SPECIFIC TO DISPLAY ####################

DISPLAY_WIDTH, DISPLAY_HEIGHT = 7, 10
# First is area, second is what to print (0=nothing, 1=ppl, 2=area ID, 3=power, 4=defense)
map_display = [
 [(0,0),(0,1),(0,4),(0,0),(6,2),(6,3),(7,0),],
 [(0,2),(0,3),(0,0),(6,1),(6,4),(7,0),(7,2),],
 [(0,0),(1,0),(1,3),(1,0),(6,0),(7,1),(7,4),],
 [(2,0),(1,0),(1,2),(1,0),(1,0),(1,0),(7,3),],
 [(2,0),(1,0),(1,1),(1,4),(5,2),(5,0),(5,3),],
 [(2,0),(2,2),(2,0),(4,0),(5,0),(5,1),(5,4),],
 [(2,0),(2,1),(2,4),(4,2),(4,3),(8,0),(8,2),],
 [(2,0),(2,3),(2,0),(4,1),(4,4),(8,1),(8,4),],
 [(3,0),(3,2),(3,0),(3,0),(3,3),(8,0),(8,3),],
 [(3,0),(3,0),(3,1),(3,4),(3,0),(3,0),(8,0),],
]

# for area in range(NB_AREAS):
# 	displayed = [map_display[y][x][1] for y in range(DISPLAY_HEIGHT) for x in range(DISPLAY_WIDTH) if map_display[y][x][0] == area and map_display[y][x][1] > 0]
# 	displayed_sorted = sorted(displayed)
# 	print(area, displayed_sorted)
# 	if len(displayed_sorted) != 4 or [i for i in range(1,5) if i not in displayed_sorted]:
# 		breakpoint()