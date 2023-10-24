from SmallworldConstants import *

NB_ROUNDS = 10

#################### MAP DESCRIPTION ####################

NB_AREAS = 30

# Area description (terrain)
#    Terrain    cavern magic  mine lost-tribe at-edge
descr = np.array([
	[ WATER   , False, False, False, False,  True ], #0
	[ MOUNTAIN, True , False, True , False,  True ], #1
	[ FARMLAND, False, True , False, False,  True ], #2
	[ HILLT   , False, False, False, False,  True ], #3
	[ FORESTT , False, False, True , False,  True ], #4
	[ FARMLAND, False, True , False, True ,  True ], #5
	[ SWAMPT  , False, False, False, False,  False], #6
	[ SWAMPT  , False, False, False, True ,  False], #7
	[ MOUNTAIN, False, False, False, False,  True ], #8
	[ MOUNTAIN, False, False, False, False,  True ], #9
	[ FORESTT , True , False, False, True ,  False], #10
	[ HILLT   , False, True , False, True ,  False], #11
	[ WATER   , False, False, False, False,  False], #12
	[ HILLT   , False, False, False, False,  False], #13
	[ MOUNTAIN, False, False, False, False,  True ], #14
	[ FARMLAND, False, False, False, False,  True ], #15
	[ MOUNTAIN, False, False, True , False,  False], #16
	[ FORESTT , False, False, False, False,  True ], #17
	[ SWAMPT  , True , False, False, False,  True ], #18
	[ MOUNTAIN, True , False, False, False,  False], #19
	[ SWAMPT  , False, True , False, True ,  False], #20
	[ FARMLAND, False, False, False, True ,  False], #21
	[ HILLT   , False, True , False, False,  False], #22
	[ FARMLAND, False, False, False, True ,  False], #23
	[ FORESTT , False, False, True , True ,  True ], #24
	[ FORESTT , False, False, False, True ,  True ], #25
	[ HILLT   , True , False, False, True ,  True ], #26
	[ WATER   , False, False, False, False,  True ], #27
	[ SWAMPT  , False, False, True , False,  True ], #28
	[ MOUNTAIN, False, False, False, False,  True ], #29
], dtype=np.int8)

# Describe which areas are neighbours
connexity_list = [
	[1, 4, 5, ],             # 0
	[0, 5, 6, 7, 2, ],       # 1
	[1, 7, 10, 8, 3, ],      # 2
	[2, 8, ],                # 3
	[0, 5, 6, 9, ],          # 4
	[0, 1, 6, 4, ],          # 5
	[4, 5, 1, 7, 11, 9],     # 6
	[1, 2, 10, 12, 11, 6],   # 7
	[3, 2, 10, 14, ],        # 8
	[4, 6, 11, 15, ],        # 9
	[7, 2, 8, 14, 13, 12],   # 10
	[9, 6, 7, 12, 16, 15],   # 11
	[11, 7, 10, 13, 20, 19, 16], # 12
	[12, 10, 14, 17, 20, ],  # 13
	[8, 10, 13, 17, ],       # 14
	[9, 11, 16, 18, ],       # 15
	[15, 11, 12, 19, 21, 18],# 16
	[14, 13, 20, 24, ],      # 17
	[15, 16, 21, 25, ],      # 18
	[16, 12, 20, 23, 22, 21],# 19
	[19, 12, 13, 17, 24, 23],# 20
	[18, 16, 19, 22, 28, 25],# 21
	[21, 19, 23, 26, 29, 28],# 22
	[19, 20, 24, 26, 22, ],  # 23
	[23, 20, 17, 27, 26, ],  # 24
	[18, 21, 28, ],          # 25
	[22, 23, 24, 27, 29, ],  # 26
	[24, 26, 29, ],          # 27
	[25, 21, 22, 29, ],      # 28
	[28, 22, 26, 27, ],      # 29
]
connexity_matrix = np.zeros((NB_AREAS, NB_AREAS), dtype=np.int8)
for i in range(NB_AREAS):
	for j in connexity_list[i]:
		connexity_matrix[i,j], connexity_matrix[j,i] = True, True

#################### SPECIFIC TO DISPLAY ####################

DISPLAY_WIDTH, DISPLAY_HEIGHT = 14, 10
# First is area id, second is what to print (0=nothing, 1=ppl, 2=area ID, 3=power, 4=defense)
# 2 3 - 1 4
map_display = [
	[(0,2), (4,1), (4,4), (4,2), (9,2) , (9,3) , (15,2), (15,3), (15,0), (18,2), (18,3), (25,2), (25,3), (25,0)],
	[(0,3), (5,2), (5,3), (4,3), (9,1) , (9,4) , (15,1), (15,4), (18,0), (18,1), (18,4), (25,1), (25,4), (28,2)],
	[(0,1), (5,1), (5,4), (6,1), (6,4) , (11,2), (11,3), (16,1), (16,4), (21,2), (21,1), (21,4), (28,1), (28,4)],
	[(0,4), (1,0), (1,0), (6,3), (6,2) , (11,1), (11,4), (16,2), (16,3), (21,3), (22,1), (22,4), (28,3), (29,1)],
	[(1,0), (1,2), (1,3), (7,2), (7,1) , (7,4) , (12,2), (12,3), (19,2), (19,3), (22,2), (22,3), (29,2), (29,3)],
	[(1,0), (1,1), (1,4), (7,3), (10,0), (12,0), (12,1), (12,4), (19,1), (19,4), (23,2), (26,2), (26,3), (29,4)],
	[(2,2), (2,3), (2,0), (2,0), (10,2), (10,3), (13,2), (12,0), (20,2), (20,3), (23,1), (23,4), (26,1), (27,0)],
	[(2,1), (2,4), (2,0), (2,0), (10,1), (10,4), (13,1), (13,4), (20,1), (20,4), (24,0), (23,3), (26,4), (27,1)],
	[(2,0), (3,2), (3,3), (8,1), (8,4) , (14,2), (14,3), (13,3), (17,2), (20,0), (24,2), (24,1), (24,4), (27,4)],
	[(3,0), (3,1), (3,4), (8,3), (8,2) , (14,1), (14,4), (17,3), (17,1), (17,4), (24,3), (27,0), (27,3), (27,2)],
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