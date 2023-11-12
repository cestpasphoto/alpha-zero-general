from .SmallworldConstants import NUMBER_PLAYERS

if NUMBER_PLAYERS == 2:
	from .SmallworldMaps_2pl import *
elif NUMBER_PLAYERS == 3:
	from .SmallworldMaps_3pl import *
elif NUMBER_PLAYERS == 4:
	from .SmallworldMaps_4pl import *
else:
	raise Exception('Number of players not supported')