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

DECLINED_SPIRIT = 0
DECLINED = 1
ACTIVE   = 2

# CONSTANTS
WATER    = 0
SAND     = 1
FARMLAND = 2
MOUNTAIN = 3
MEADOW   = 4
FORESTT  = 5
HILLT    = 6
SWAMPT   = 6

NOPOWR = 0
SOURCE = 1
MINE   = 2
MAGIC  = 3
CAVERN = 4

#################### PEOPLE AND POWERS ####################

NOPPL     = 0
AMAZON    = 1  #  +4 pour attaque                                         DONE
DWARF     = 2  #  +1 victoire sur mine, même en déclin                    DONE
ELF       = 3  #  pas de défausse lors d'une défaite                      DONE
GHOUL     = 4  #  tous les zombies restent en déclin, peuvent attaquer    DONE
GIANT     = 5  #  -1 pour attaque voisin montagne                         DONE
HALFLING  = 6  #  départ n'importe où, immunité sur 2 prem régions        DONE
HUMAN     = 7  #  +1 victoire sur champs                                  DONE
ORC       = 8  #  +1 victoire pour région non-vide conquise               DONE
RATMAN    = 9  #  leur nombre                                             
SKELETON  = 10 #  +1 pion pour toutes 2 régions non-vide conquises        DONE
SORCERER  = 11 #  remplace pion unique adversaire actif par un sorcier    DONE
TRITON    = 12 #  -1 pour attaque région côtière                          DONE
TROLL     = 13 #  +1 défense sur chaque territoire même en déclin         DONE
WIZARD    = 14 #  +1 victoire sur source magique                          DONE
LOST_TRIBE= 15

MAX_SKELETONS = 20
MAX_SORCERERS = 18
#                       1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
initial_nb_people = [0, 6, 3, 6, 5, 6, 6, 5, 5, 8, 6, 5, 6, 5, 5, 2]
initial_tokens    = [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]



NOPOWER     = 0
ALCHEMIST   = 1  # +2 chaque tour      												DONE
BERSERK     = 2  # Lancer de dé AVANT chaque attaque
BIVOUACKING = 3  # 5 défenses à placer à chaque tour + immunité au sorcier          
COMMANDO    = 4  # -1 attaque        												DONE !
DIPLOMAT    = 5  # Paix avec un peuple actif à choisir à chaque tour
DRAGONMASTER= 6  # 1 attaque dragon par tour + immunité complète
FLYING      = 7  # Toutes les régions sont voisines                                 DONE !
FOREST      = 8  # +1 victoire si forêt                                             DONE
FORTIFIED   = 9  # +1 défense avec forteresse mm en déclin, +1 par tour actif (max 6)
HEROIC      = 10 # 2 immunités complètes                                            
HILL        = 11 # +1 victoire par colline                                          DONE
MERCHANT    = 12 # +1 victoire par région                                           DONE !
MOUNTED     = 13 # -1 attaque colline/ferme                                         DONE
PILLAGING   = 14 # +1 par région non vide conquise                                  DONE !
SEAFARING   = 15 # Conquête possible des mers/lacs, conservées en déclin
SPIRIT      = 16 # 2e peuple en déclin, et le reste jusqu'au bout
STOUT       = 17 # Déclin possible juste après tour classique
SWAMP       = 18 # +1 victoire par marais                                           DONE
UNDERWORLD  = 19 # -1 attaque caverne, et les cavernes sont adjacentes              DONE
WEALTHY     = 20 # +7 victoire à la fin premier tour								DONE !
#                        1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
initial_nb_power   = [0, 4, 4, 5, 4, 5, 5, 5, 4, 3, 5, 4, 2, 5, 5, 5, 5, 4, 4, 5, 4]
initial_tokens_pwr = [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7]

PHASE_READY          = 1 # Next action is to play
PHASE_CHOOSE         = 2 # Chose
PHASE_ABANDON        = 3 # Abandon
PHASE_CONQUEST       = 4 # Include preparation, attack, abandon, specialppl
PHASE_CONQ_WITH_DICE = 5 # Dice (not in berserk case)
PHASE_REDEPLOY       = 6 # Include redeploy, specialpower
PHASE_WAIT           = 7 # End of turn (after redeploy, or decline)

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
#  Terrain    power   edge   lost-tribe 
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