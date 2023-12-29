import numpy as np

NUMBER_PLAYERS = 2 # Will automatically switch to adequate map

############################# TERRAIN CONSTANTS #############################

# Some terrains names have a leading 't' to differenciate with power
FORESTT  = 0
FARMLAND = 1
HILLT    = 2
SWAMPT   = 3
MOUNTAIN = 4
WATER    = 5

NOPOWERT = 0
CAVERN   = 1
MAGIC    = 2
MINE     = 3

############################# GAME CONSTANTS #############################

DICE_VALUES = np.array([0, 0, 0, 1, 2, 3], dtype=np.int8)
AVG_DICE = 1
MAX_DICE = 3

DECK_SIZE = 6
SCORE_INIT = 5
SCORE_OFFSET = 128
IMMUNITY = 20
MAX_REDEPLOY = 8

DECLINED_SPIRIT = 0
DECLINED = 1
ACTIVE   = 2

PHASE_READY            = 1 # Next action is to play
PHASE_CHOOSE           = 2 # Chose
PHASE_ABANDON          = 3 # Abandon
PHASE_CONQUEST         = 4 # Include preparation, attack, abandon, specialppl
PHASE_CONQ_WITH_DICE   = 5 # Dice (not in berserk case)
PHASE_ABANDON_AMAZONS  = 6 # Forced to give back some amazons (no more attack allowed)
PHASE_REDEPLOY         = 7 # Include redeploy, specialpower
PHASE_STOUT_TO_DECLINE = 8 # Going to decline for stout (count score now), temporary status
PHASE_WAIT             = 9 # End of turn (after redeploy, or decline)

############################# PEOPLE CONSTANTS #############################

NOPPL     = 0
AMAZON    = 1  #  +4 pour attaque
DWARF     = 2  #  +1 victoire sur mine, même en déclin
ELF       = 3  #  pas de défausse lors d'une défaite
GHOUL     = 4  #  tous les zombies restent en déclin, peuvent attaquer
GIANT     = 5  #  -1 pour attaque voisin montagne
HALFLING  = 6  #  départ n'importe où, immunité sur 2 prem régions
HUMAN     = 7  #  +1 victoire sur champs
ORC       = 8  #  +1 victoire pour région non-vide conquise
RATMAN    = 9  #  leur nombre                                             
SKELETON  = 10 #  +1 pion pour toutes 2 régions non-vide conquises
SORCERER  = 11 #  remplace pion unique adversaire actif par un sorcier
TRITON    = 12 #  -1 pour attaque région côtière
TROLL     = 13 #  +1 défense sur chaque territoire même en déclin
WIZARD    = 14 #  +1 victoire sur source magique
LOST_TRIBE=-15

MAX_SKELETONS = 20
MAX_SORCERERS = 18
#                                1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
initial_nb_people = np.array([0, 6, 3, 6, 5, 6, 6, 5, 5, 8, 6, 5, 6, 5, 5, 1], dtype=np.int8)
initial_tokens    = np.array([0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)


NOPOWER     = 0
ALCHEMIST   = 1  # +2 chaque tour
BERSERK     = 2  # Lancer de dé AVANT chaque attaque
BIVOUACKING = 3  # 5 défenses à placer à chaque tour + immunité au sorcier
COMMANDO    = 4  # -1 attaque
DIPLOMAT    = 5  # Paix avec un peuple actif à choisir à chaque tour
DRAGONMASTER= 6  # 1 attaque dragon par tour + immunité complète
FLYING      = 7  # Toutes les régions sont voisines
FOREST      = 8  # +1 victoire si forêt
FORTIFIED   = 9  # +1 défense avec forteresse mm en déclin, +1 par tour actif (max 6- doit limiter à +une fortress / tour
HEROIC      = 10 # 2 immunités complètes
HILL        = 11 # +1 victoire par colline
MERCHANT    = 12 # +1 victoire par région
MOUNTED     = 13 # -1 attaque colline/ferme
PILLAGING   = 14 # +1 par région non vide conquise
SEAFARING   = 15 # Conquête possible des mers/lacs, conservées en déclin
SPIRIT      = 16 # 2e peuple en déclin, et le reste jusqu'au bout
STOUT       = 17 # Déclin possible juste après tour classique
SWAMP       = 18 # +1 victoire par marais
UNDERWORLD  = 19 # -1 attaque caverne, et les cavernes sont adjacentes
WEALTHY     = 20 # +7 victoire à la fin premier tour
#                                 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
initial_nb_power   = np.array([0, 4, 4, 5, 4, 5, 5, 5, 4, 3, 5, 4, 2, 5, 5, 5, 5, 4, 4, 5, 4], dtype=np.int8)
initial_tokens_pwr = np.array([0, 0, 0, 5, 0, 0, 0, 0, 0, 6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7], dtype=np.int8)
