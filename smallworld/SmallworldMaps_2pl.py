from SmallworldConstants import *

NB_ROUNDS = 10

#################### MAP DESCRIPTION ####################

NB_AREAS = 23

# Area description (terrain)
#    Terrain    cavern magic  mine lost-tribe at-edge
descr = np.array([
	[ WATER   , False, False, False, False,  True ], #0
	[ MOUNTAIN, True , False, True , False,  True ], #1
	[ FARMLAND, False, False, False, True ,  True ], #2
	[ SWAMPT  , False, True , False, True ,  True ], #3
	[ FARMLAND, False, True , False, False,  True ], #4
	[ HILLT   , False, False, False, True ,  False], #5
	[ HILLT   , True , False, False, False,  True ], #6
	[ FORESTT , False, False, False, True ,  False], #7
	[ FORESTT , False, False, True , False,  True ], #8
	[ WATER   , False, False, False, False,  False], #9
	[ SWAMPT  , False, False, True , True ,  True ], #10
	[ MOUNTAIN, False, False, False, False,  False], #11
	[ FARMLAND, False, True , False, True ,  False], #12
	[ SWAMPT  , True , False, False, True ,  True ], #13
	[ MOUNTAIN, False, False, False, False,  True ], #14
	[ HILLT   , True , False, False, True ,  False], #15
	[ FARMLAND, False, False, False, False,  False], #16
	[ SWAMPT  , False, False, False, False,  True ], #17
	[ HILLT   , False, False, False, False,  True ], #18
	[ WATER   , False, False, False, False,  True ], #19
	[ MOUNTAIN, False, False, True , False,  True ], #20
	[ FORESTT , False, False, False, False,  True ], #21
	[ FORESTT , False, True , False, True ,  True ], #22
], dtype=np.int8)

# Describe which areas are neighbours
connexity_list = [
	[1, 4],                      # 0
	[0, 4, 5, 2],                # 1
	[1, 5, 7, 6, 3],             # 2
	[2, 6],                      # 3
	[0, 1, 5, 8],                # 4
	[4, 1, 2, 7, 9, 8],          # 5
	[3, 2, 7, 10],               # 6
	[5, 2, 6, 10, 12, 9],        # 7
	[4, 5, 9, 11, 13],           # 8
	[8, 5, 7, 12, 11],           # 9
	[6, 7, 12, 14],              # 10
	[8, 9, 12, 15, 16, 13],      # 11
	[11, 9, 7, 10, 14, 17, 15],  # 12
	[8, 11, 16, 18],             # 13
	[10, 12, 17],                # 14
	[16, 11, 12, 17, 21, 20, 16],# 15
	[13, 11, 15, 22, 18],        # 16
	[14, 12, 15, 21, 19],        # 17
	[13, 16, 22],                # 18
	[17, 21, 20],                # 19
	[22, 15, 21, 19],            # 20
	[20, 15, 17, 19],            # 21
	[18, 16, 15, 20],            # 22
]
connexity_matrix = np.zeros((NB_AREAS, NB_AREAS), dtype=np.int8)
for i in range(NB_AREAS):
	for j in connexity_list[i]:
		connexity_matrix[i,j], connexity_matrix[j,i] = True, True

#################### SPECIFIC TO DISPLAY ####################

DISPLAY_WIDTH, DISPLAY_HEIGHT = 13, 8
# First is area id, second is what to print (0=nothing, 1=ppl, 2=area ID, 3=power, 4=defense)
# 2 3 - 1 4
map_display = [
	[(0,2), (0,3), (4,2), (4,3), (8,2) , (8,3) , (13,1), (13,4), (13,3), (18,2), (18,3), (18,1), (18,4)],
	[(0,1), (0,4), (4,1), (4,4), (8,1) , (11,0), (11,2), (13,2), (16,2), (16,1), (16,4), (22,2), (22,3)],
	[(1,2), (1,3), (1,0), (5,2), (8,4) , (9,3) , (11,1), (11,4), (16,3), (15,0), (22,0), (22,1), (22,4)],
	[(1,1), (1,4), (5,1), (5,4), (9,1) , (9,4) , (11,3), (15,0), (15,2), (15,3), (20,0), (20,2), (20,3)],
	[(2,2), (2,3), (5,3), (7,2), (7,3) , (9,2) , (12,2), (12,3), (15,1), (15,4), (20,0), (20,1), (20,4)],
	[(2,1), (2,4), (2,0), (7,1), (7,4) , (12,0), (12,1), (12,4), (17,2), (17,3), (21,2), (21,3), (20,0)],
	[(3,2), (3,3), (6,2), (6,3), (10,2), (10,3), (14,2), (14,3), (17,1), (17,4), (21,1), (21,4), (19,3)],
	[(3,1), (3,4), (6,1), (6,4), (10,1), (10,4), (14,1), (14,4), (17,0), (19,0), (19,1), (19,4), (19,2)],
]


def check_map():
	assert(not np.any(np.transpose(connexity_matrix) - connexity_matrix))

	assert(len(map_display) == DISPLAY_HEIGHT)
	assert(len(map_display[0]) == DISPLAY_WIDTH)

	for area in range(NB_AREAS):
		displayed = [map_display[y][x][1] for y in range(DISPLAY_HEIGHT) for x in range(DISPLAY_WIDTH) if map_display[y][x][0] == area and map_display[y][x][1] > 0]
		displayed_sorted = sorted(displayed)
		print(area, displayed_sorted)
		if len(displayed_sorted) != 4 or [i for i in range(1,5) if i not in displayed_sorted]:
			breakpoint()

# check_map()