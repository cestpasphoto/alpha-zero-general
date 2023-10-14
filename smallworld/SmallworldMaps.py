from SmallworldConstants import *

#################### MAP DESCRIPTION ####################

NB_AREAS = 9

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

# Area description (terrain)
#  Terrain    power    edge   lost-tribe 
descr = [
 [ WATER    , NOPOWERT, True  ,  False], #0
 [ SWAMPT   , MINE    , False ,  False], #1
 [ FARMLAND , CAVERN  , True  ,  True ], #2
 [ WATER    , NOPOWERT, True  ,  False], #3
 [ MOUNTAIN , NOPOWERT, False ,  False], #4
 [ FARMLAND , CAVERN  , True  ,  True ], #5
 [ FORESTT  , NOPOWERT, True  ,  False], #6
 [ MOUNTAIN , MINE    , True  ,  True ], #7
 [ HILLT    , MAGIC   , True  ,  True ], #8
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

def check_map():
	assert(not np.any(np.transpose(connexity_matrix) - connexity_matrix))

	for area in range(NB_AREAS):
		displayed = [map_display[y][x][1] for y in range(DISPLAY_HEIGHT) for x in range(DISPLAY_WIDTH) if map_display[y][x][0] == area and map_display[y][x][1] > 0]
		displayed_sorted = sorted(displayed)
		print(area, displayed_sorted)
		if len(displayed_sorted) != 4 or [i for i in range(1,5) if i not in displayed_sorted]:
			breakpoint()