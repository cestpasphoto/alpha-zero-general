from .SmallworldConstants import *

NB_ROUNDS = 9

#################### MAP DESCRIPTION ####################

NB_AREAS = 39

# Area description (terrain)
#    Terrain    cavern magic  mine lost-tribe at-edge
descr = np.array([
	[WATER     , False, False, False, False, True ],  #  0
	[MOUNTAIN  , False, False, False, False, True ],  #  1
	[SWAMPT    , False, False, True , True , True ],  #  2
	[FARMLAND  , False, False, False, False, True ],  #  3
	[FARMLAND  , False, False, False, False, True ],  #  4
	[SWAMPT    , False, True , False, True , True ],  #  5
	[FORESTT   , False, False, False, False, True ],  #  6
	[MOUNTAIN  , False, False, False, False, True ],  #  7
	[HILLT     , True , False, False, True , True ],  #  8
	[SWAMPT    , False, False, False, True , False],  #  9
	[HILLT     , False, False, False, False, False],  # 10
	[FORESTT   , False, True , False, True , False],  # 11
	[HILLT     , False, False, True , False, True ],  # 12
	[FARMLAND  , False, False, True , True , True ],  # 13
	[MOUNTAIN  , True , False, True , False, False],  # 14
	[FORESTT   , True , False, False, True , True ],  # 15
	[HILLT     , False, True , False, False, False],  # 16
	[SWAMPT    , True , False, False, True , False],  # 17
	[FARMLAND  , False, False, False, False, False],  # 18
	[MOUNTAIN  , False, False, False, False, False],  # 19
	[FARMLAND  , False, False, False, False, True ],  # 20
	[HILLT     , False, False, False, True , True ],  # 21
	[WATER     , False, False, False, False, False],  # 22
	[FARMLAND  , False, False, False, True , False],  # 23
	[FORESTT   , False, True , False, True , False],  # 24
	[MOUNTAIN  , False, False, True , False, False],  # 25
	[FORESTT   , False, True , False, True , False],  # 26
	[SWAMPT    , False, False, True , False, False],  # 27
	[MOUNTAIN  , False, False, True , False, True ],  # 28
	[SWAMPT    , False, True , False, False, True ],  # 29
	[FORESTT   , True , False, False, False, True ],  # 30
	[HILLT     , False, False, False, True , False],  # 31
	[MOUNTAIN  , True , False, False, False, False],  # 32
	[MOUNTAIN  , False, False, False, False, True ],  # 33
	[FARMLAND  , False, False, False, False, True ],  # 34
	[FORESTT   , False, False, False, False, True ],  # 35
	[SWAMPT    , True , False, False, True , True ],  # 36
	[HILLT     , False, True , False, False, True ],  # 37
	[WATER     , False, False, False, False, True ],  # 38
], dtype=np.int8)

# Describe which areas are neighbours
connexity_list = [
	[1,	6, 5, 4, 8, 13, 15],          # 0
	[0, 6, 10, 2],                    # 1
	[1, 10, 11, 3],                   # 2
	[2, 11, 7],                       # 3
	[0, 5],                           # 4
	[4, 0, 8, 6],                     # 5
	[5, 9, 10, 1, 0, 8],              # 6
	[3, 11, 17, 12],                  # 7
	[0, 13, 9, 6, 5, 18],             # 8
	[8, 18, 16, 14, 10, 6],           # 9
	[9, 14, 11, 2, 1, 6],             #10
	[10, 14, 19, 17, 7, 3, 2],        #11
	[7, 17, 20],                      #12
	[0, 15, 21, 18, 8],               #13
	[16, 22, 19, 11, 10, 9],          #14
	[21, 13, 0],                      #15
	[18, 22, 14, 9],                  #16
	[19, 23, 24, 20, 12, 7, 11],      #17
	[13, 21, 25, 22, 16, 9, 8],       #18
	[22, 23, 17, 11, 14],             #19
	[24, 28, 12, 17],                 #20
	[29, 25, 18, 13, 15],             #21
	[25, 26, 27, 23, 19, 14, 16, 18], #22
	[27, 31, 32, 28, 24, 17, 19, 22], #23
	[23, 28, 20, 17],                 #24
	[21, 30, 26, 22, 18, 29],         #25
	[30, 33, 27, 22, 25],             #26
	[26, 33, 34, 31, 23, 22],         #27
	[23, 32, 35, 20, 24],             #28
	[21, 30, 25],                     #29
	[29, 33, 26, 25],                 #30
	[27, 34, 36, 37, 32, 23],         #31
	[31, 37, 35, 28, 23],             #32
	[30, 34, 27, 26],                 #33
	[33, 36, 31, 27],                 #34
	[37, 38, 28, 32],                 #35
	[34, 37, 31],                     #36
	[36, 38, 35, 32, 31],             #37
	[35, 37],                         #38
]
connexity_matrix = np.zeros((NB_AREAS, NB_AREAS), dtype=np.int8)
for i in range(NB_AREAS):
	for j in connexity_list[i]:
		connexity_matrix[i,j], connexity_matrix[j,i] = True, True

#################### SPECIFIC TO DISPLAY ####################

DISPLAY_WIDTH, DISPLAY_HEIGHT = 14, 13
# First is area id, second is what to print (0=nothing, 1=ppl, 2=area ID, 3=power, 4=defense)
# 2 3 - 1 4
map_display = [
	[(0,2), (0,0), (0,0), (0,0) , (0,0) , (15,3), (15,1), (15,4), (21,2), (21,0), (21,1), (21,4), (29,2), (29,0),],
	[(0,3), (4,1), (4,4), (0,0) , (13,2), (13,3), (15,2), (21,0), (21,3), (29,0), (29,0), (29,3), (29,1), (29,4),],
	[(0,0), (4,2), (5,3), (0,0) , (8,2) , (13,1), (13,4), (21,0), (25,2), (25,3), (30,2), (30,3), (30,1), (30,4),],
	[(0,1), (5,1), (5,4), (8,1) , (8,4) , (18,2), (18,3), (18,4), (25,1), (25,4), (26,2), (26,3), (33,2), (33,3),],
	[(0,4), (5,2), (6,2), (8,3) , (9,2) , (9,3) , (16,2), (18,1), (22,2), (26,1), (26,4), (33,0), (33,1), (33,4),],
	[(0,0), (0,0), (6,3), (9,1) , (9,4) , (16,3), (16,1), (16,4), (22,3), (27,2), (27,3), (27,0), (34,1), (34,4),],
	[(1,0), (6,1), (6,4), (10,2), (14,0), (14,2), (14,3), (22,0), (22,1), (27,1), (27,4), (34,2), (34,3), (36,2),],
	[(1,0), (1,2), (1,3), (10,1), (10,4), (14,1), (14,4), (19,2), (22,4), (23,0), (31,2), (31,3), (36,1), (36,4),],
	[(1,1), (1,4), (2,2), (10,3), (11,2), (11,3), (19,3), (19,1), (19,4), (23,4), (31,1), (31,4), (37,2), (36,3),],
	[(2,1), (2,4), (2,3), (11,0), (11,1), (11,4), (17,2), (23,1), (23,2), (23,3), (32,2), (32,3), (37,1), (37,4),],
	[(2,0), (3,0), (3,0), (11,0), (7,2) , (17,1), (17,4), (24,2), (24,3), (28,2), (32,1), (32,4), (35,3), (37,3),],
	[(3,2), (3,3), (7,1), (7,4) , (7,3) , (12,2), (17,3), (24,1), (24,4), (28,1), (28,4), (35,1), (35,4), (38,2),],
	[(3,1), (3,4), (7,0), (12,1), (12,4), (12,3), (20,2), (20,3), (20,1), (20,4), (28,3), (35,2), (38,1), (38,4),],
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