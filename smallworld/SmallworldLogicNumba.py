import numpy as np
from numba import njit
import numba

from .SmallworldConstants import *
from .SmallworldMaps import *
from .SmallworldDisplay import print_board, print_valids, move_to_str

USERANDOM = True

############################## BOARD DESCRIPTION ##############################

# Board is described by a NB_AREAS+5*NUMBER_PLAYERS+7 x 8 array
# (40x5 for 2 players for instance, 40 = 23 + 5*2 + 7)
# Most rows contain the generic info using the following template:
#     0      Number of people
#     1      Type of people, negative if in decline
#     2      Special power, negative if in decline
# Here is the description of each line of the board. For readibility, we defined
# "shortcuts" that actually are views (numpy name) of overal board.
##### Index  Shortcut              Subindex	 Meaning
#####   0    self.territories        0-2     Generic info (nb, type, power) for territory (or area) n°0
#####                                 3      Defense/immunity due to type of people, in same territory
#####                                 4      Defense/immunity due to special power, in same territory
#####                                 5      Total defense of territory
#####                                 6      Points of territory if owner scored now (0 if no owner)
#####                                 7      Player who owns this territory (-1 if no owner)
#####  0-22  self.territories        0-4     Same, in territory i
#####   23   self.peoples (3D)     0   0-2   Generic info (nb, type, power) for declined+spirit people in player 0's hand
#####                              0    3    Various info related to people capacity for declined+spirit people in player 0's hand
#####                              0    4    Various info related to special power for declined+spirit people in player 0's hand
#####                              0    5    -
#####                              0    6    Additional points due to this people if player scored now (0 if NOPPL)
#####                              0    7    Player id (0)
#####   24                         1   0-4   Same for declined (non-spirit) people in player 0's hand
#####   25                         2   0-4   Same for active people in player 0's hand
#####  26-28                      0-2  0-4   Same info for peoples in player 1's hand
#####  29-34 self.visible_deck       0-2     Generic info (nb, type, power) for each combo on deck
#####                                3-5     - empty -
#####                                 6      Number of victory points to win with such combo
#####                                 7      - empty (-1) -
#####  35-36 self.round_status        0      Total number of people on the map for player i
#####                                1-2     - empty -
#####                                 3      #NETWDT = number of Non-Empty Territories Won During Turn
#####                                 4      Status of people (from PHASE_READ to PHASE_WAIT)
#####                                 5      Total defense of people on the map
#####                                 6      Additional points if player i scored now
#####                                 7      Player id (i)
#####  37-38 self.game_status        0-2     - empty - 
#####                                 3      Round of next move of player i, from 1 to 10
#####                                 4      Id of people current playing (from DECLINED_SPIRIT to ACTIVE), -1 else
#####                                 5      - empty -
#####                                 6      Total score of player i
#####                                 7      Player id (i)
#####   39   self.invisible_deck     0-1     Bitfield stating if people #i (0-14) is already in deck or used
#####                                2-4     Bitfield stating if power #i (0-20) is already in deck or used
#####                                 5      Number of dice usage
#####                                 6      Number of random usage for deck
#####                                 7      - empty -
# Indexes above are assuming 2 players, you can have more details in copy_state().
#
# How is used self.peoples[:,:,3] depending on people:
#  Amazon  : number of people under "loan", 4 during attack and should be 0 at end of turn
#  Skeleton: 1 if their power was applied this turn
#  Halfling: number of used holes-in-ground
#  Sorcerer: bitfield of which players have been sorcerized during this turn
#            (0 means current player, 1 means next player, etc)
#
# How is used self.peoples[:,:,4] depending on power:
#  Diplomat: during turn   = bitfield of people who player attacked + 2^6 to
#                            differentiate with next case
#            between turns = relative ID of people who diplomacy applies to
#                            (0 = no diplomacy)
#            Use relative ID: 0 means current player, 1 means next player, etc
#  Bivouacking, fortified, heroic: number of bonus "defense" left

############################## ACTION DESCRIPTION #############################
# We coded 131 actions, taking some shortcuts on combinations of gems that can be
# got or that can be given back, and forbidding to simultaneously get gems and
# give some back.
# Here is description of each action:
##### Index   Meaning
#####   0-22  Abandon
#####  23-45  Attack
#####  46-68  Special action using people capacity
#####  69-91  Special action using power
#####   92    Skip redeploy (no more redeploy)
#####  93-100 Redeploy N ppl on each area
##### 100-122 Redeploy 1 ppl on area N
##### 123-128 Choose a people from deck
#####   129   Decline
#####   130   End

@njit(cache=True, fastmath=True, nogil=True)
def observation_size():
	return (NB_AREAS + 5*NUMBER_PLAYERS + DECK_SIZE+1, 8)

@njit(cache=True, fastmath=True, nogil=True)
def action_size():
	return 5*NB_AREAS + MAX_REDEPLOY + DECK_SIZE + 2

@njit(cache=True, fastmath=True, nogil=True)
def my_dot(array1, array2):
	return np.multiply(array1, array2).sum(dtype=np.int8)

@njit(cache=True, fastmath=True, nogil=True)
def my_dot2d(array1, array2):
	return np.multiply(array1, array2).sum(axis=1, dtype=np.int8)

mask = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint16)

@njit(cache=True, fastmath=True, nogil=True)
def my_packbits(array):
	if len(array)%8 == 0:
		padded_array = array.astype(np.uint16)
	else:
		padded_array = np.append(array, np.zeros(8-len(array)%8)).astype(np.uint16)
	result = my_dot2d(padded_array.reshape((-1, 8)), mask)
	return result

@njit(cache=True, fastmath=True, nogil=True)
def my_unpackbits(values):
	result = np.zeros((len(values), 8), dtype=np.uint8)
	for i, v in enumerate(values):
		result[i, :] = (np.bitwise_and(v, mask) != 0)
	return result.flatten()

@njit(cache=True, fastmath=True, nogil=True)
def _split_pwr_data(unified_value):
	dataB, dataA = divmod(unified_value, 2**6)
	dataB = bool(dataB)
	return dataA, dataB

spec = [
	('state'         , numba.int8[:,:]),
	('territories'   , numba.int8[:,:]),
	('peoples'       , numba.int8[:,:,:]),
	('visible_deck'  , numba.int8[:,:]),
	('round_status'  , numba.int8[:,:]),
	('game_status'   , numba.int8[:,:]),
	('invisible_deck', numba.int8[:]),
]
@numba.experimental.jitclass(spec)
class Board():
	def __init__(self, num_players):
		self.state = np.zeros(observation_size(), dtype=np.int8)
		self.init_game()

	def get_score(self, player):
		return self.game_status[player, 6] + SCORE_OFFSET

	def init_game(self):
		self.copy_state(np.zeros(observation_size(), dtype=np.int8), copy_or_not=False)

		# Fill map with lost tribe
		nb_lt = initial_nb_people[-LOST_TRIBE]
		for i in range(NB_AREAS):
			self.territories[i,:] = [nb_lt, LOST_TRIBE, NOPOWER, 0, 0, nb_lt+int(descr[i, 0] == MOUNTAIN), 0, -1] if descr[i][4] else [0, NOPPL, NOPOWER, 0, 0, 0, 0, -1]

		# Init deck
		self._init_deck()

		# Init money and status for each player
		self.round_status[0, 4] = PHASE_READY
		self.round_status[1:, 4] = PHASE_WAIT
		self.round_status[:, 7] = np.arange(NUMBER_PLAYERS)
		self.game_status[0, 4] = ACTIVE
		self.game_status[1:, 4] = -1
		self.game_status[:, 6] = SCORE_INIT - SCORE_OFFSET
		self.game_status[:, 7] = np.arange(NUMBER_PLAYERS)

		for ppl_id in range(ACTIVE+1):
			self.peoples[:, ppl_id, 7] = np.arange(NUMBER_PLAYERS)
		
		# First round is round #1
		self._update_round()
	
	def copy_state(self, state, copy_or_not):
		if self.state is state and not copy_or_not:
			return
		self.state = state.copy() if copy_or_not else state

		self.territories    = self.state[0                                  :NB_AREAS                             ,:]
		self_peoples_2d     = self.state[NB_AREAS                           :NB_AREAS+3*NUMBER_PLAYERS            ,:]
		self.peoples = np.ascontiguousarray(self_peoples_2d).reshape((NUMBER_PLAYERS, 3, 8))
		self.visible_deck   = self.state[NB_AREAS+3*NUMBER_PLAYERS          :NB_AREAS+3*NUMBER_PLAYERS+DECK_SIZE  ,:]
		self.round_status   = self.state[NB_AREAS+3*NUMBER_PLAYERS+DECK_SIZE:NB_AREAS+4*NUMBER_PLAYERS+DECK_SIZE  ,:]
		self.game_status    = self.state[NB_AREAS+4*NUMBER_PLAYERS+DECK_SIZE:NB_AREAS+5*NUMBER_PLAYERS+DECK_SIZE  ,:]
		self.invisible_deck = self.state[NB_AREAS+5*NUMBER_PLAYERS+DECK_SIZE                                      ,:]

	def valid_moves(self, player):
		result = np.zeros(action_size(), dtype=np.bool_)
		result[                       :  NB_AREAS]              = self._valids_abandon(player)
		result[  NB_AREAS             :2*NB_AREAS]              = self._valids_attack(player)
		result[2*NB_AREAS             :3*NB_AREAS]              = self._valids_special_actionppl(player)
		result[3*NB_AREAS             :4*NB_AREAS]              = self._valids_special_actionpwr(player)
		result[4*NB_AREAS             :5*NB_AREAS+MAX_REDEPLOY] = self._valids_redeploy(player)
		result[5*NB_AREAS+MAX_REDEPLOY:5*NB_AREAS+MAX_REDEPLOY+DECK_SIZE] = self._valids_choose_ppl(player)
		result[5*NB_AREAS+MAX_REDEPLOY+DECK_SIZE]               = self._valid_decline(player)
		result[5*NB_AREAS+MAX_REDEPLOY+DECK_SIZE+1]             = self._valid_end(player)
							
		return result

	def make_move(self, move, player, deterministic):
		current_ppl, _ = self._current_ppl(player)

		if   move < NB_AREAS:
			area = move
			self._do_abandon(player, area, deterministic)
		elif move < 2*NB_AREAS:
			area = move - NB_AREAS
			self._do_attack(player, area, deterministic)
		elif move < 3*NB_AREAS:
			area = move - 2*NB_AREAS
			self._do_special_actionppl(player, area, deterministic)
		elif move < 4*NB_AREAS:
			area = move - 3*NB_AREAS
			self._do_special_actionpwr(player, area, deterministic)
		elif move < 5*NB_AREAS+MAX_REDEPLOY:
			param = move - 4*NB_AREAS
			self._do_redeploy(player, param, deterministic)
		elif move < 5*NB_AREAS+MAX_REDEPLOY+DECK_SIZE:
			area = move - 5*NB_AREAS-MAX_REDEPLOY
			self._do_choose_ppl(player, area, deterministic)	
		elif move < 5*NB_AREAS+MAX_REDEPLOY+DECK_SIZE+1:
			self._do_decline(player, deterministic)
		elif move < 5*NB_AREAS+MAX_REDEPLOY+DECK_SIZE+2:
			self._do_end(player, deterministic)
		else:
			print('Unknown move {move}')

		if self.game_status[player, 4] >= 0:
			return player
		return (player+1)%NUMBER_PLAYERS

	def get_state(self):
		return self.state

	def get_round(self):
		return self.game_status[:, 3].min()

	def check_end_game(self, next_player):
		if self.get_round() <= NB_ROUNDS:
			return np.full(NUMBER_PLAYERS, 0., dtype=np.float32) # No winner yet

		# Game is ended
		scores = self.game_status[:, 6]
		best_score = scores.max()
		several_winners = ((scores == best_score).sum() > 1)
		return np.where(scores == best_score, 0.01 if several_winners else 1., -1.).astype(np.float32)

	# if nb_swaps=1, transform P0 to Pn, P1 to P0, ... and Pn to Pn-1
	# else do this action n times
	def swap_players(self, nb_swaps):
		def _roll_in_place_axis0_2d(array):
			tmp_copy = array.copy()
			for i in range(array.shape[0]):
				array[i,:7] = tmp_copy[(i+nb_swaps)%NUMBER_PLAYERS,:7]
		def _roll_in_place_axis0_3d(array):
			tmp_copy = array.copy()
			for i in range(array.shape[0]):
				array[i,:,:7] = tmp_copy[(i+nb_swaps)%NUMBER_PLAYERS,:,:7]
		def _roll_in_place_territories(territories):
			for area in range(territories.shape[0]):
				if territories[area, 7] < 0:
					continue
				territories[area, 7] = (territories[area, 7] - nb_swaps) % NUMBER_PLAYERS

		# "Roll" peoples and status
		_roll_in_place_territories(self.territories)
		_roll_in_place_axis0_2d(self.round_status)
		_roll_in_place_axis0_2d(self.game_status)
		_roll_in_place_axis0_3d(self.peoples)

	def get_symmetries(self, policy, valids):
		# Always called on canonical board, meaning player = 0
		symmetries = [(self.state.copy(), policy.copy(), valids.copy())]
		state_backup, policy_backup, valids_backup = symmetries[0]

		# Whatever the score difference is
		scores = self.game_status[:, 6]
		for _ in range(2):
			mini, maxi = -127-scores.min(), 127-scores.max()
			if mini >= maxi:
				print('symmetrie scores:', mini, maxi)
			else:
				random_offset = np.random.randint(mini, maxi)
				self.game_status[:, 6] += random_offset
				symmetries.append((self.state.copy(), policy.copy(), valids.copy()))
				self.state[:,:], policy, valids = state_backup.copy(), policy_backup.copy(), valids_backup.copy()

		# # (Approximate symmetry, remove when NN is strong)
		# # Declined people all have same effects (with some exceptions), we can swap
		# blacklist_ppl = [DWARF, GHOUL, TROLL]
		# available_people = my_unpackbits(self.invisible_deck[0:2])
		# if available_people.sum() != 0:
		# 	for ppl in blacklist_ppl:
		# 		available_people[ppl] = False
		# 	for p in range(NUMBER_PLAYERS):
		# 		declined_ppl_type = self.peoples[p, DECLINED, 1]
		# 		if declined_ppl_type == NOPPL or -declined_ppl_type in blacklist_ppl:
		# 			continue
		# 		# Swap type with another random one
		# 		new_ppl_type = -np.random.choice(np.flatnonzero(available_people))
		# 		for area in range(NB_AREAS):
		# 			if self.territories[area, 1] == declined_ppl_type:
		# 				self.territories[area, 1] = new_ppl_type
		# 		self.peoples[p, DECLINED, 1] = new_ppl_type

		# 		symmetries.append((self.state.copy(), policy.copy(), valids.copy()))
		# 		self.state[:,:], policy, valids = state_backup.copy(), policy_backup.copy(), valids_backup.copy()

		# # (Approximate symmetry, remove when NN is strong)
		# # Having n declined ppl in hand is like having none at all, unless ghoul
		# for p in range(NUMBER_PLAYERS):
		# 	for ppl_id in [DECLINED, DECLINED_SPIRIT]:
		# 		if self.peoples[p, ppl_id, 0] > 0 and self.peoples[p, ppl_id, 1] != -GHOUL:
		# 			self.peoples[p, ppl_id, 0] = 0

		# 			symmetries.append((self.state.copy(), policy.copy(), valids.copy()))
		# 			self.state[:,:], policy, valids = state_backup.copy(), policy_backup.copy(), valids_backup.copy()

		# # (Approximate symmetry, remove when NN is strong)
		# # For non-playing peoples, having n active ppl in hand is like having none at all
		# for p in range(1, NUMBER_PLAYERS):
		# 	if self.peoples[p, ACTIVE, 0] > 0:
		# 		self.peoples[p, ACTIVE, 0] = 0

		# 		symmetries.append((self.state.copy(), policy.copy(), valids.copy()))
		# 		self.state[:,:], policy, valids = state_backup.copy(), policy_backup.copy(), valids_backup.copy()

		return symmetries

	###########################################################################

	def _valids_attack(self, player):
		valids = np.zeros(NB_AREAS, dtype=np.bool_)
		current_ppl, current_id = self._current_ppl(player)

		# Check that player has a valid ppl:
		if current_ppl[1] == NOPPL:
			return valids

		# Attack permitted when it's the time
		if self.round_status[player, 4] not in [PHASE_READY, PHASE_CHOOSE, PHASE_ABANDON, PHASE_CONQUEST]:
			return valids

		territories_of_player = self._are_occupied_by(current_ppl)
		how_many_ppl_available = self._ppl_virtually_available(player, current_ppl, PHASE_CONQUEST, territories_of_player)
		# Forbid to continue if 0 ppl left in hand
		if how_many_ppl_available <= 0:
			return valids

		# Take in account dice when berserk
		if current_ppl[2] == BERSERK:
			if _split_pwr_data(current_ppl[4])[1]:
				how_many_ppl_available += _split_pwr_data(current_ppl[4])[0]

		# Compute list of conceivable areas to be checked further
		# Area must not be current people
		conditions = np.logical_not(territories_of_player)
		# Area must have no immunity
		conditions = np.logical_and(conditions, self.territories[:, 5] < IMMUNITY)
		# Area must no be water
		if current_ppl[2] != SEAFARING:
			conditions = np.logical_and(conditions, descr[:, 0] != WATER)
		# Area must be close to owned areas, or on the edge if first conquest
		if current_ppl[2] != FLYING:
			if np.count_nonzero(territories_of_player) == 0:
				if current_ppl[1] != HALFLING:
					conditions = np.logical_and(conditions, descr[:, 5] != 0)
			else:
				# neighbor_areas = np.logical_or.reduce(connexity_matrix[territories_of_player])
				neighbor_areas = (connexity_matrix[territories_of_player].sum(axis=0) != 0)
				if current_ppl[2] == UNDERWORLD:
					# All caverns are neighbors for underworlds
					if np.any(np.logical_and(descr[:,CAVERN], territories_of_player)):
						neighbor_areas = np.logical_or(neighbor_areas, descr[:, CAVERN])
				conditions = np.logical_and(conditions, neighbor_areas)

		conceivable_areas = np.flatnonzero(conditions)
		for area in conceivable_areas:
			valids[area] = self._valid_attack_area(player, area, current_ppl, how_many_ppl_available, territories_of_player)

		return valids

	def _valid_attack_area(self, player, area, current_ppl, how_many_ppl_available, territories_of_player):
		# Check that player has a chance to win	(no 2nd dice for berserk)
		minimum_ppl_for_attack = self._minimum_ppl_for_attack(area, current_ppl)
		if how_many_ppl_available + (0 if current_ppl[2] == BERSERK else MAX_DICE) < minimum_ppl_for_attack:
			return False
		
		# Check that active ppl is not attacking a diplomat in peace
		if self.territories[area, 2] == DIPLOMAT and current_ppl[1] > 0:
			loser_ppl, loser_id = self._ppl_owner_of(area)
			if loser_ppl[4] == (player - loser_id) % NUMBER_PLAYERS:
				return False

		return True

	def _do_attack(self, player, area, deterministic):
		current_ppl, current_id = self._current_ppl(player)

		# If 1st action of the turn, prepare people
		self._prepare_for_new_status(player, current_ppl, PHASE_CONQUEST, deterministic)

		nb_ppl_of_player, minimum_ppl_for_attack = current_ppl[0],self._minimum_ppl_for_attack(area, current_ppl)

		# Use dice if people are needed
		use_dice = (nb_ppl_of_player < minimum_ppl_for_attack)
		if current_ppl[2] == BERSERK and _split_pwr_data(current_ppl[4])[1]:
			dice = _split_pwr_data(current_ppl[4])[0]
			if nb_ppl_of_player + dice < minimum_ppl_for_attack:
				self.round_status[player, 4] = PHASE_CONQ_WITH_DICE
				return
			nb_attacking_ppl = max(minimum_ppl_for_attack - dice, 1)
		elif use_dice:
			if USERANDOM:
				if deterministic == 0:
					dice = np.random.choice(DICE_VALUES)
				else:
					# https://stackoverflow.com/questions/3062746/special-simple-random-number-generator
					# m=6, c=5, a=1980+1
					rnd_value = (1981 * (deterministic+self.invisible_deck[5]) + 5) % 6
					dice = DICE_VALUES[rnd_value]
				self.invisible_deck[5] += 1
			else:
				dice = DICE_VALUES[3]
			if nb_ppl_of_player + dice < minimum_ppl_for_attack:
				self.round_status[player, 4] = PHASE_CONQ_WITH_DICE
				return
			nb_attacking_ppl = nb_ppl_of_player
		else:
			nb_attacking_ppl = minimum_ppl_for_attack

		# Attack is successful

		# Update loser and winner
		self._switch_territory_from_loser_to_winner(area, player, current_ppl, nb_attacking_ppl)

		# Deal with berserk AFTER attack
		if current_ppl[2] == BERSERK:
			self._switch_status_berserk(player, current_ppl, None, PHASE_CONQUEST, deterministic)
		# Update winner's status
		self.round_status[player, 4] = PHASE_CONQ_WITH_DICE if use_dice else PHASE_CONQUEST
		self._update_round_status(current_ppl, player)

	def _valids_redeploy(self, player):
		valids = np.zeros(NB_AREAS + MAX_REDEPLOY, dtype=np.bool_)
		current_ppl, current_id = self._current_ppl(player)

		# Check that player has a valid ppl:
		if current_ppl[1] == NOPPL:
			return valids

		# Check that it is time
		if self.round_status[player, 4] in [PHASE_WAIT, PHASE_ABANDON_AMAZONS]:
			return valids

		# Check there is at least one active territory
		territories_of_player = self._are_occupied_by(current_ppl)
		nb_territories = np.count_nonzero(territories_of_player)
		if nb_territories == 0:
			if self.round_status[player, 4] != PHASE_REDEPLOY:
				valids[0] = True # If no other option, then allow to skip redeploy
			return valids

		# Check that player has still some ppl to deploy
		how_many_ppl_available = self._ppl_virtually_available(player, current_ppl, PHASE_REDEPLOY, territories_of_player)
		if how_many_ppl_available == 0:
			if self.round_status[player, 4] != PHASE_REDEPLOY:
				valids[0] = True # If no other option, then allow to skip redeploy
			return valids
		elif how_many_ppl_available < 0:
			return valids    # Redeploy not allowed, need to abandon instead

		for ppl_to_deploy in range(1, MAX_REDEPLOY):
			valids[ppl_to_deploy] = (how_many_ppl_available >= ppl_to_deploy * nb_territories)
		for area in range(NB_AREAS):
			valids[MAX_REDEPLOY + area] = territories_of_player[area]

		if not valids.any() and self.round_status[player, 4] != PHASE_REDEPLOY:
			valids[0] = True # If no other option, then allow to skip redeploy

		return valids

	def _do_redeploy(self, player, param, deterministic):
		current_ppl, current_id = self._current_ppl(player)

		if param == 0: # Special case, skip redeploy
			self._prepare_for_new_status(player, current_ppl, PHASE_REDEPLOY, deterministic)
			self.round_status[player, 4] = PHASE_REDEPLOY
			# Status already changed by previous function
			self._update_round_status(current_ppl, player)
			self._end_turn_if_possible(player, current_ppl, deterministic)
			return

		self._prepare_for_new_status(player, current_ppl, PHASE_REDEPLOY, deterministic)
		self.round_status[player, 4] = PHASE_REDEPLOY

		if param < MAX_REDEPLOY:
			# Deploy X ppl on all active areas
			how_many_to_deploy = param
			territories_of_player = self._are_occupied_by(current_ppl)
			current_ppl[0] -= how_many_to_deploy * np.count_nonzero(territories_of_player)
			self.territories[:, 0] += how_many_to_deploy*territories_of_player
			self.territories[:, 5] += how_many_to_deploy*territories_of_player

		else:
			# Deploy 1 ppl on 1 area
			area = param - MAX_REDEPLOY
			current_ppl[0]            -= 1
			self.territories[area, 0] += 1
			self.territories[area, 5] += 1

		self._update_round_status(current_ppl, player)
		self._end_turn_if_possible(player, current_ppl, deterministic)

	def _valid_decline(self, player):
		# Going to decline permitted only for active_ppl
		if self.game_status[player, 4] != ACTIVE or self.peoples[player, ACTIVE, 1] == NOPPL:
			return False
		# Going to decline permitted only on 1st move (except for stout)
		if self.round_status[player, 4] != PHASE_READY:
			if self.round_status[player, 4] in [PHASE_CONQUEST, PHASE_CONQ_WITH_DICE, PHASE_REDEPLOY] and self.peoples[player, ACTIVE, 2] == STOUT:
				pass # Exception
			else:
				return False
		return True

	def _do_decline(self, player, deterministic):
		current_ppl = self.peoples[player, ACTIVE, :]

		# Count score now for stout
		if current_ppl[2] == STOUT:
			self._prepare_for_new_status(player, current_ppl, PHASE_STOUT_TO_DECLINE, deterministic)
			self.round_status[player, 4] = PHASE_STOUT_TO_DECLINE

		declined_id = DECLINED_SPIRIT if current_ppl[2] == SPIRIT else DECLINED

		# If we are going to replace declined ppl...
		if self.peoples[player, declined_id, 1] != NOPPL:
			# Remove previous declined ppl from the board
			for area in range(NB_AREAS):
				if self._is_occupied_by(area, self.peoples[player, declined_id, :]):
					self.territories[area] = [0, NOPPL, NOPOWER, 0, 0, 0, 0, -1]

			self.peoples[player, declined_id, :7] = 0
			self._update_deck_after_decline()

		# Move ppl to decline and keep only 1 ppl per territory except if ghoul
		if current_ppl[1] == GHOUL:
			self.peoples[player, declined_id, 0] = current_ppl[0]
		else:
			self._gather_current_ppl_but_one(current_ppl)
		self.peoples[player, declined_id, 1] = current_ppl[1]
		current_ppl[:7] = 0
		
		# Flip back ppl tokens on the board and remove defense
		for area in range(NB_AREAS):
			if self.territories[area, 1] == self.peoples[player, declined_id, 1]:
				backup = self.territories[area, :].copy()
				self.territories[area, 1] = -self.peoples[player, declined_id, 1]
				# Remove defense, except some cases
				self.territories[area, 2:7] = 0
				if backup[2] == FORTIFIED:
					self.territories[area, 4] = backup[4]
				self._update_territory_after_win_or_decline(current_ppl, player, area)

		self.peoples[player, declined_id, 1:3] = -self.peoples[player, declined_id, 1:3]

		# Count score and switch to next player depending on current status		
		self._update_round_status(self.peoples[player, declined_id, :], player)
		self._prepare_for_new_status(player, current_ppl, PHASE_WAIT, deterministic)
		self.round_status[player, 4] = PHASE_WAIT

	def _valids_choose_ppl(self, player):
		valids = np.zeros(DECK_SIZE, dtype=np.bool_)

		# Check that it is time
		if self.round_status[player, 4] != PHASE_READY:
			return valids
		# Check not declined ppl
		if self.game_status[player, 4] != ACTIVE:
			return valids
		# Check that player hasn't a player yet
		if self.peoples[player, ACTIVE, 1] != NOPPL:
			return valids

		for index in range(DECK_SIZE):
			# Check that index is valid and player can pay
			valids[index] = (self.visible_deck[index, 1] != NOPPL) and (self.game_status[player, 6] + SCORE_OFFSET >= index)

		return valids

	def _do_choose_ppl(self, player, index, deterministic):
		current_ppl = self.peoples[player, ACTIVE, :]

		current_ppl[:3] = self.visible_deck[index, :3]
		current_ppl[3]  = initial_tokens[current_ppl[1]]
		current_ppl[4]  = initial_tokens_pwr[current_ppl[2]]
		current_ppl[5:7]= 0

		# Earn money but also pay what's needed
		self.game_status[player, 6] += self.visible_deck[index, 6] - index

		self._prepare_for_new_status(player, current_ppl, PHASE_CHOOSE, deterministic)
		self.round_status[player, 4] = PHASE_CHOOSE
		self._update_deck_after_chose(index, deterministic)

	def _valids_abandon(self, player):
		valids = np.zeros(NB_AREAS, dtype=np.bool_)
		current_ppl, current_id = self._current_ppl(player)

		if self.round_status[player, 4] not in [PHASE_READY, PHASE_ABANDON, PHASE_ABANDON_AMAZONS]:
			if current_ppl[1] == AMAZON and self.round_status[player, 4] in [PHASE_CONQUEST, PHASE_CONQ_WITH_DICE] and self._ppl_virtually_available(player, current_ppl, PHASE_REDEPLOY) < 0:
				pass # exception if Amazons can't redeploy
			else:
				return valids
		# Cant abandon if player doesn't have any active people
		if current_ppl[1] == NOPPL:
			return valids

		for area in range(NB_AREAS):
			valids[area] = self._is_occupied_by(area, current_ppl)

		return valids

	def _do_abandon(self, player, area, deterministic):
		current_ppl, current_id = self._current_ppl(player)
		self._leave_area(area)
		if self.round_status[player, 4] in [PHASE_CONQUEST, PHASE_CONQ_WITH_DICE, PHASE_ABANDON_AMAZONS]:
			# exception if Amazons abandoned because couldn't redeploy
			if self._ppl_virtually_available(player, current_ppl, PHASE_REDEPLOY) >= 0:
				self._prepare_for_new_status(player, current_ppl, PHASE_REDEPLOY, deterministic)
				self.round_status[player, 4] = PHASE_REDEPLOY
			else:
				self._prepare_for_new_status(player, current_ppl, PHASE_ABANDON_AMAZONS, deterministic)
				self.round_status[player, 4] = PHASE_ABANDON_AMAZONS
		else:
			self._prepare_for_new_status(player, current_ppl, PHASE_ABANDON, deterministic)
			self.round_status[player, 4] = PHASE_ABANDON

		self._update_round_status(current_ppl, player)

	def _valids_special_actionppl(self, player):
		valids = np.zeros(NB_AREAS, dtype=np.bool_)
		current_ppl, current_id = self._current_ppl(player)

		if current_ppl[1] == SORCERER:
			# Attack permitted when it's the time
			if self.round_status[player, 4] not in [PHASE_READY, PHASE_CHOOSE, PHASE_ABANDON, PHASE_CONQUEST]:
				return valids

			# Limited number of sorcerers in the box
			territories_of_player = self._are_occupied_by(current_ppl)
			total_number = self._total_number_of_ppl(current_ppl, territories_of_player)
			if total_number + 1 > MAX_SORCERERS:
				return valids
			
			for area in range(NB_AREAS):
				valids[area] = self._valid_special_actionppl_area(player, area, current_ppl, territories_of_player)

		return valids

	def _valid_special_actionppl_area(self, player, area, current_ppl, territories_of_player):
		if current_ppl[1] == SORCERER:
			# No attack on water
			if descr[area][0] == WATER and current_ppl[2] != SEAFARING:
				return False
			# People on this area is alone and active
			if self.territories[area, 0] != 1 or self.territories[area, 1] <= 0:
				return False
			# No attack on current people
			if self._is_occupied_by(area, current_ppl):
				return False
			# Check no full immunity
			if self.territories[area, 3] >= IMMUNITY or self.territories[area, 4] >= IMMUNITY:
				return False
			# Check that territory is close to another owned territory or is on the edge (unless is flying)
			if current_ppl[2] != FLYING:
				neighbor_areas = connexity_matrix[area]
				if not np.any(np.logical_and(neighbor_areas, territories_of_player)):
					return False
			# Check that opponent had not been already 'sorcerized' during this turn
			loser_ppl, loser = self._ppl_owner_of(area)
			if current_ppl[3] & 2**( (player-loser)%NUMBER_PLAYERS ):
				return False
			# Check that opponent isn't protected by a campment
			if loser_ppl[2] == BIVOUACKING and self.territories[area, 4] > 0:
				return False

			return True

		else:
			return False

	def _do_special_actionppl(self, player, area, deterministic):
		current_ppl, current_id = self._current_ppl(player)

		if current_ppl[1] == SORCERER:
			loser_ppl, loser = self._ppl_owner_of(area)
			# Prepare people if 1st action of the turn
			self._prepare_for_new_status(player, current_ppl, PHASE_CONQUEST, deterministic)
			# Replace
			self.territories[area,:] = [1, SORCERER, current_ppl[2], 0, 0, 0, 0, player]
			# Note that loser have been 'sorcerized'
			current_ppl[3] |= 2**( (player-loser)%NUMBER_PLAYERS )
			# Update winner's status and #NETWDT
			self.round_status[player, 4] = PHASE_CONQUEST
			self.round_status[player, 3] += 1

			self._update_territory_after_win_or_decline(loser_ppl, loser, area)
			self._update_territory_after_win_or_decline(current_ppl, player, area)
			self._update_round_status(current_ppl, player)
		else:
			raise Exception('Should not happen')

	def _valids_special_actionpwr(self, player):
		valids = np.zeros(NB_AREAS, dtype=np.bool_)
		current_ppl, current_id = self._current_ppl(player)

		if current_ppl[2] == BIVOUACKING:
			# Place campments when it's the time
			if self.round_status[player, 4] not in [PHASE_CONQUEST, PHASE_CONQ_WITH_DICE, PHASE_REDEPLOY]:
				return valids

			# Limited number of campments in the box
			if current_ppl[4] <= 0:
				return valids

			# Check there are enough amazons
			if not self._enough_amazons_to_redeploy(player, current_ppl):
				return valids
			
			for area in range(NB_AREAS):
				valids[area] = self._valid_special_actionpwr_area(player, area, current_ppl)
		
		elif current_ppl[2] == FORTIFIED:
			# Place fortress when it's the time
			if self.round_status[player, 4] not in [PHASE_CONQUEST, PHASE_CONQ_WITH_DICE, PHASE_REDEPLOY]:
				return valids

			# Check if any fortress remaining and if no other fortress used in this turn
			remaining_fort, used_fort = _split_pwr_data(current_ppl[4])
			if remaining_fort <= 0 or used_fort:
				return valids

			# Check there are enough amazons
			if not self._enough_amazons_to_redeploy(player, current_ppl):
				return valids
			
			for area in range(NB_AREAS):
				valids[area] = self._valid_special_actionpwr_area(player, area, current_ppl)

		elif current_ppl[2] == HEROIC:
			# Place hero when it's the time
			if self.round_status[player, 4] not in [PHASE_CONQUEST, PHASE_CONQ_WITH_DICE, PHASE_REDEPLOY]:
				return valids

			# Check if any hero remaining
			if current_ppl[4] <= 0:
				return valids

			# Check there are enough amazons
			if not self._enough_amazons_to_redeploy(player, current_ppl):
				return valids
			
			for area in range(NB_AREAS):
				valids[area] = self._valid_special_actionpwr_area(player, area, current_ppl)

		elif current_ppl[2] == DIPLOMAT:
			# Chose when it's the time
			if self.round_status[player, 4] not in [PHASE_CONQUEST, PHASE_CONQ_WITH_DICE]:
				return valids

			# Check there are enough amazons
			if not self._enough_amazons_to_redeploy(player, current_ppl):
				return valids

			for other_player in range(NUMBER_PLAYERS):
				valids[other_player] = self._valid_special_actionpwr_area(player, other_player, current_ppl)

		elif current_ppl[2] == DRAGONMASTER:
			# Attack permitted when it's the time
			if self.round_status[player, 4] not in [PHASE_READY, PHASE_CHOOSE, PHASE_ABANDON, PHASE_CONQUEST]:
				return valids

			# Check 1st time this power is used during turn
			if current_ppl[4] > 0:
				return valids

			# Check at least 1 ppl remaining
			if current_ppl[0] < 1:
				return valids

			for area in range(NB_AREAS):
				valids[area] = self._valid_special_actionpwr_area(player, area, current_ppl)

		return valids

	def _valid_special_actionpwr_area(self, player, area, current_ppl):
		if current_ppl[2] == BIVOUACKING:
			# Apply only on own territories
			if not self._is_occupied_by(area, current_ppl):
				return False
			return True

		elif current_ppl[2] == FORTIFIED:
			# Apply only on own territories
			if not self._is_occupied_by(area, current_ppl):
				return False
			# If there is no fortress already
			if self.territories[area, 4] > 0:
				return False
			return True

		elif current_ppl[2] == HEROIC:
			# Apply only on own territories
			if not self._is_occupied_by(area, current_ppl):
				return False
			# If there is no hero already
			if self.territories[area, 4] > 0:
				return False
			return True

		elif current_ppl[2] == DIPLOMAT:
			# Param is actually id of other player minus id of current player
			other_player_relative_id = area
			# Check not attacked during this turn
			if current_ppl[4] & 2**other_player_relative_id:
				return False
			return True

		elif current_ppl[2] == DRAGONMASTER:
			# No attack on water
			if descr[area][0] == WATER:
				return False
			# No attack on current people
			territories_of_player = self._are_occupied_by(current_ppl)
			if territories_of_player[area]:
				return False
			# Check no full immunity
			if self.territories[area, 3] >= IMMUNITY or self.territories[area, 4] >= IMMUNITY:
				return False
			# Check that territory is close to another owned territory or is on the edge
			neighbor_areas = connexity_matrix[area]
			if not np.any(np.logical_and(neighbor_areas, territories_of_player)):
				return False
			return True

		else:
			return False

	def _do_special_actionpwr(self, player, area, deterministic):
		current_ppl, current_id = self._current_ppl(player)

		if current_ppl[2] == BIVOUACKING:
			# Put campment
			self.territories[area, 4] += 1
			self.territories[area, 5] += 1
			current_ppl[4]            -= 1

			self._prepare_for_new_status(player, current_ppl, PHASE_REDEPLOY, deterministic)
			self.round_status[player, 4] = PHASE_REDEPLOY
			self._update_round_status(current_ppl, player)

		elif current_ppl[2] == FORTIFIED:
			# Put fortress
			self.territories[area, 4] += 1
			self.territories[area, 5] += 1
			self.territories[area, 6] += 1
			current_ppl[4]            -= 1
			# Note that we used a fortress during this turn
			current_ppl[4]            |= 2**6

			self._prepare_for_new_status(player, current_ppl, PHASE_REDEPLOY, deterministic)
			self.round_status[player, 4] = PHASE_REDEPLOY
			self._update_round_status(current_ppl, player)

		elif current_ppl[2] == HEROIC:
			# Put hero
			self.territories[area, 5] += (IMMUNITY - self.territories[area, 4])
			self.territories[area, 4]  = IMMUNITY
			current_ppl[4]            -= 1

			self._prepare_for_new_status(player, current_ppl, PHASE_REDEPLOY, deterministic)
			self.round_status[player, 4] = PHASE_REDEPLOY
			self._update_round_status(current_ppl, player)

		elif current_ppl[2] == DIPLOMAT:
			other_player_relative_id = area
			# Set diplomacy
			current_ppl[4] = other_player_relative_id

			self._prepare_for_new_status(player, current_ppl, PHASE_REDEPLOY, deterministic)
			self.round_status[player, 4] = PHASE_REDEPLOY

		elif current_ppl[2] == DRAGONMASTER:
			# Remove previous dragon
			for area_ in self._are_occupied_by(current_ppl).nonzero()[0]:
				if self.territories[area_, 4] != 0:
					self.territories[area_, 5] -= self.territories[area_, 4]
					self.territories[area_, 4] = 0
			self._prepare_for_new_status(player, current_ppl, PHASE_CONQUEST, deterministic)
			
			# Update loser and winner, and add dragon
			self._switch_territory_from_loser_to_winner(area, player, current_ppl, nb_attacking_ppl=1)
			self.territories[area, 5] += IMMUNITY
			self.territories[area, 4]  = IMMUNITY
			# Note that we used the power
			current_ppl[4] = 1
			self.round_status[player, 4] = PHASE_CONQUEST

			self._update_round_status(current_ppl, player)

		else:
			raise Exception('Should not happen')

	def _valid_end(self, player):
		current_ppl, current_id = self._current_ppl(player)
		return self._valid_end_aux(player, current_ppl)

	def _valid_end_aux(self, player, current_ppl):
		# Check it's time
		if self.round_status[player, 4] != PHASE_REDEPLOY:
			return False
		if current_ppl[1] == NOPPL:
			return False
		# Check no more people in hand
		if current_ppl[0] > 0 and np.count_nonzero(self._are_occupied_by(current_ppl)) > 0:
			if current_ppl[1] == AMAZON and current_ppl[0] == current_ppl[3]:
				pass
			else:
				return False

		# Check that enough amazon to give back
		if not self._enough_amazons_to_redeploy(player, current_ppl):
			return False

		return True

	def _do_end(self, player, deterministic):
		current_ppl, current_id = self._current_ppl(player)
		self._update_round_status(current_ppl, player)
		self._prepare_for_new_status(player, current_ppl, PHASE_WAIT, deterministic)
		# Status change already handled by previous function

	###########################################################################

	def _current_ppl(self, player):
		current_id = self.game_status[player, 4]
		if current_id < 0:
			raise Exception('No ppl to play for P{player}')
		return self.peoples[player, current_id, :], current_id

	def _ppl_owner_of(self, area):
		if self.territories[area, 1] == NOPPL or self.territories[area, 1] == LOST_TRIBE:
			return None, -1
		result = np.argwhere(self.peoples[:,:,1] == self.territories[area, 1])
		if result.shape[0] != 1:
			raise Exception('Could not find which ppl this area belongs ({area=} {self.territories[area, 1]=} {result=})')
		return self.peoples[result[0][0], result[0][1], :], result[0][0]

	def _is_occupied_by(self, area, current_ppl):
		return (self.territories[area,1] == current_ppl[1])

	def _are_occupied_by(self, current_ppl):
		return (self.territories[:,1] == current_ppl[1])

	def _is_area_border_of(self, area, terrain):
		neighbor_areas = connexity_matrix[area]
		areas_with_terrain = (descr[:,0] == terrain)
		result = np.any(np.logical_and(neighbor_areas, areas_with_terrain)).item()
		return result

	def _minimum_ppl_for_attack(self, area, current_ppl):
		minimum_ppl_for_attack = self.territories[area, 5] + 2

		# Bonus if: triton + border of water, giant + border of mountain, commando,
		#   mounted + hill|farm, underworld + cavern, 
		if current_ppl[1] == TRITON and self._is_area_border_of(area, WATER):
			minimum_ppl_for_attack -= 1
		if current_ppl[1] == GIANT  and self._is_area_border_of(area, MOUNTAIN):
			minimum_ppl_for_attack -= 1
		if current_ppl[2] == COMMANDO:
			minimum_ppl_for_attack -= 1
		if current_ppl[2] == MOUNTED and descr[area][0] in [HILLT, FARMLAND]:
			minimum_ppl_for_attack -= 1
		if current_ppl[2] == UNDERWORLD and descr[area][CAVERN]:
			minimum_ppl_for_attack -= 1

		return max(minimum_ppl_for_attack, 1)

	def _leave_area(self, area):
		leaver_ppl, leaver_id = self._ppl_owner_of(area)

		# Give back ppl to owner, and tokens if needed
		leaver_ppl[0] += self.territories[area, 0]
		if self.territories[area, 2] in [BIVOUACKING, FORTIFIED]:
			leaver_ppl[4] += self.territories[area, 4]
		elif self.territories[area, 2] == HEROIC and self.territories[area, 4] > 0:
			leaver_ppl[4] += 1

		# Make the area empty
		self.territories[area,:7] = 0
		self.territories[area,7] = -1

	def _switch_territory_from_loser_to_winner(self, area, player, winner_ppl, nb_attacking_ppl):
		nb_initial_ppl = self.territories[area, 0]

		# Give back people to the loser (if any)
		loser_ppl, loser_id = self._ppl_owner_of(area)
		if loser_ppl is not None:
			# Lose 1 ppl unless you are elf and active
			nb_ppl_to_lose = 1 if self.territories[area,1] != ELF else 0
			loser_ppl[0] += self.territories[area,0] - nb_ppl_to_lose
			# Give back tokens
			if self.territories[area, 2] in [BIVOUACKING, FORTIFIED]:
				loser_ppl[4] += self.territories[area, 4]
			elif self.territories[area, 2] == HEROIC and self.territories[area, 4] > 0:
				loser_ppl[4] += 1
			# Note successful attack if diplomat
			if winner_ppl[2] == DIPLOMAT:
				winner_ppl[4] |= 2**( (player-loser_id)%NUMBER_PLAYERS )

		# Install people from the winner
		self.territories[area,0] = nb_attacking_ppl
		self.territories[area,1:3] = winner_ppl[1:3]
		self.territories[area,3:7] = 0
		self.territories[area,7] = player
		winner_ppl[0] -= nb_attacking_ppl

		if loser_ppl is not None:
			self._update_round_status(loser_ppl, loser_id)
		self._update_territory_after_win_or_decline(winner_ppl, player, area)

		# Update #NETWDT
		if nb_initial_ppl > 0:
			self.round_status[player, 3] += 1

	def _total_number_of_ppl(self, current_ppl, player_territories=None):
		if player_territories is None:
			player_territories = self._are_occupied_by(current_ppl)

		how_many_on_board = my_dot(self.territories[:,0], player_territories)
		how_many_in_hand  = current_ppl[0]
		return how_many_on_board + how_many_in_hand

	def _limit_added_ppl(self, current_ppl, addition, maximum, player_territories=None):
		total_number = self._total_number_of_ppl(current_ppl, player_territories)
		return min(addition, maximum - total_number)

	def _gather_current_ppl_but_one(self, current_ppl):
		# Gather all active people in player's hand, leaving only 1 on each territory
		for area in range(NB_AREAS):
			if self._is_occupied_by(area, current_ppl):
				nb_ppl_to_gather = max(self.territories[area,0] - 1, 0)
				if nb_ppl_to_gather > 0:
					self.territories[area,0] -= nb_ppl_to_gather
					self.territories[area,5] -= nb_ppl_to_gather
					current_ppl[0]           += nb_ppl_to_gather

	# All changes in this function must be reported in _prepare_for_ready(x,y) = _prepare_for_new_status(x,y,PHASE_READY)
	def _prepare_for_new_status(self, player, current_ppl, next_status, deterministic):
		old_status = self.round_status[player, 4]

		if old_status in [PHASE_READY] and next_status in [PHASE_ABANDON, PHASE_CONQUEST, PHASE_CONQ_WITH_DICE]:
			self._gather_current_ppl_but_one(current_ppl)
		elif old_status in [PHASE_READY, PHASE_CONQUEST, PHASE_CONQ_WITH_DICE, PHASE_ABANDON_AMAZONS] and next_status == PHASE_REDEPLOY:
			self._gather_current_ppl_but_one(current_ppl)

		# People
		if current_ppl[1] == AMAZON:
			self._switch_status_amazon(player, current_ppl, old_status, next_status)
		elif current_ppl[1] == SKELETON:
			self._switch_status_skeleton(player, current_ppl, old_status, next_status)	

		# Power
		if current_ppl[2] == BIVOUACKING:
			self._switch_status_bivouacking(player, current_ppl, old_status, next_status)
		elif current_ppl[2] == HEROIC:
			self._switch_status_heroic(player, current_ppl, old_status, next_status)
		elif current_ppl[2] == DIPLOMAT:
			self._switch_status_diplomat(player, current_ppl, old_status, next_status)
		elif current_ppl[2] == BERSERK:
			if next_status == PHASE_CONQUEST:
				pass # special case if during attack, don't prerun dice yet
			else:
				self._switch_status_berserk(player, current_ppl, old_status, next_status, deterministic)

		# For stout, compure score BEFORE going to decline, but not switching yet to next player
		if next_status == PHASE_STOUT_TO_DECLINE:
			if current_ppl[2] == STOUT:
				self._compute_and_update_score(player)

		if next_status == PHASE_WAIT:
			if self.game_status[player, 4] == ACTIVE and old_status != PHASE_STOUT_TO_DECLINE:
				self._compute_and_update_score(player)
			self._switch_to_next(player, current_ppl, deterministic)

	# Exactly same function as above but needed anyway since Numba doesn't allow recursion in _switch_to_next()
	def _prepare_for_ready(self, player, current_ppl, deterministic):
		old_status, next_status = self.round_status[player, 4], PHASE_READY

		# People
		if current_ppl[1] == AMAZON:
			self._switch_status_amazon(player, current_ppl, old_status, next_status)
		elif current_ppl[1] == SKELETON:
			self._switch_status_skeleton(player, current_ppl, old_status, next_status)	

		# Power
		if current_ppl[2] == BIVOUACKING:
			self._switch_status_bivouacking(player, current_ppl, old_status, next_status)
		elif current_ppl[2] == HEROIC:
			self._switch_status_heroic(player, current_ppl, old_status, next_status)
		elif current_ppl[2] == DIPLOMAT:
			self._switch_status_diplomat(player, current_ppl, old_status, next_status)
		elif current_ppl[2] == BERSERK:
			self._switch_status_berserk(player, current_ppl, old_status, next_status, deterministic)

	def _end_turn_if_possible(self, player, current_ppl, deterministic):
		# Still people left to deploy
		if current_ppl[0] > 0:
			return False

		# Stout may want to decline
		if current_ppl[2] == STOUT:
			return False

		# May want to put another bivouack, fortification or hero (if any left)
		if current_ppl[2] in [BIVOUACKING, FORTIFIED, HEROIC] and current_ppl[4] > 0:
			return False

		# Check proper phase and non-negative number of amazons
		if not self._valid_end_aux(player, current_ppl):
			return False

		self._do_end(player, deterministic)
		return True

	def _switch_status_amazon(self, player, current_ppl, old_status, next_status):
		if old_status in [PHASE_CONQUEST, PHASE_CONQ_WITH_DICE, PHASE_ABANDON_AMAZONS] and next_status == PHASE_REDEPLOY:
			if current_ppl[3] != 0:
				current_ppl[0] -= current_ppl[3]
				current_ppl[3]  = 0

		elif old_status in [PHASE_READY, PHASE_CHOOSE, PHASE_ABANDON] and next_status == PHASE_CONQUEST:
			if current_ppl[3] == 0:
				current_ppl[0] += 4
				current_ppl[3]  = 4

	def _switch_status_skeleton(self, player, current_ppl, old_status, next_status):
		if old_status in [PHASE_READY, PHASE_CHOOSE, PHASE_ABANDON, PHASE_CONQUEST, PHASE_CONQ_WITH_DICE, PHASE_ABANDON_AMAZONS] and next_status == PHASE_REDEPLOY:
			if current_ppl[3] == 0:
				current_ppl[0] += self._limit_added_ppl(current_ppl, self.round_status[player, 3] // 2, MAX_SKELETONS)
				current_ppl[3] = 1 # We write that we gave additional ppl already

	def _switch_status_bivouacking(self, player, current_ppl, old_status, next_status):
		if old_status in [PHASE_READY, PHASE_CHOOSE, PHASE_ABANDON] and next_status == PHASE_CONQUEST:
			# Gather back all campments
			for area in self._are_occupied_by(current_ppl).nonzero()[0]:
				if self.territories[area, 4] > 0:
					current_ppl[4] += self.territories[area, 4]
					self.territories[area, 5] -= self.territories[area, 4]
					self.territories[area, 4] = 0

	def _switch_status_heroic(self, player, current_ppl, old_status, next_status):
		if old_status in [PHASE_READY, PHASE_CHOOSE, PHASE_ABANDON] and next_status == PHASE_CONQUEST:
			# Gather back heros
			for area in self._are_occupied_by(current_ppl).nonzero()[0]:
				if self.territories[area, 4] > 0:
					current_ppl[4] += 1
					self.territories[area, 5] -= self.territories[area, 4]
					self.territories[area, 4] = 0

	def _switch_status_diplomat(self, player, current_ppl, old_status, next_status):
		if old_status in [PHASE_READY, PHASE_CHOOSE, PHASE_ABANDON] and next_status == PHASE_CONQUEST:
			# Start noting which players I attacked
			current_ppl[4] = 2**6
		elif old_status != PHASE_WAIT and next_status == PHASE_WAIT:
			# If no player chose, set peace with ... self meaning set to 0
			if _split_pwr_data(current_ppl[4])[1]:
				current_ppl[4] = 0

	def _switch_status_berserk(self, player, current_ppl, old_status, next_status, deterministic):
		if next_status in [PHASE_READY, PHASE_ABANDON, PHASE_CHOOSE, PHASE_CONQUEST]:
			# pre-run dice
			if USERANDOM:
				if deterministic == 0:
					dice = np.random.choice(DICE_VALUES)
				else:
					# https://stackoverflow.com/questions/3062746/special-simple-random-number-generator
					# m=6, c=5, a=1980+1
					rnd_value = (1981 * (deterministic+self.invisible_deck[5]) + 5) % 6
					dice = DICE_VALUES[rnd_value]
				self.invisible_deck[5] += 1
			else:
				dice = DICE_VALUES[3]
			current_ppl[4] = dice + 2**6
		else:
			current_ppl[4] = 0

	def _ppl_virtually_available(self, player, current_ppl, next_status, player_territories=None):
		if player_territories is None:
			player_territories = self._are_occupied_by(current_ppl)

		old_status = self.round_status[player, 4]
		how_many_ppl_available = current_ppl[0]
		if old_status in [PHASE_READY] and next_status in [PHASE_ABANDON, PHASE_CONQUEST, PHASE_CONQ_WITH_DICE]:
			# Simulate redeploy: add people on the boards, except 1 per territory
			how_many_ppl_available += my_dot(np.maximum(self.territories[:,0]-1,0), player_territories)
		elif old_status in [PHASE_READY, PHASE_ABANDON, PHASE_CONQUEST, PHASE_CONQ_WITH_DICE, PHASE_ABANDON_AMAZONS] and next_status == PHASE_REDEPLOY:
			# Simulate redeploy: add people on the boards, except 1 per territory
			how_many_ppl_available += my_dot(np.maximum(self.territories[:,0]-1,0), player_territories)

		# Additional people depending on people and game status
		if current_ppl[1] == AMAZON:
			if old_status in [PHASE_CONQUEST, PHASE_CONQ_WITH_DICE, PHASE_ABANDON_AMAZONS] and next_status == PHASE_REDEPLOY:
				if current_ppl[3] != 0:
					how_many_ppl_available -= current_ppl[3]

			elif old_status in [PHASE_READY, PHASE_CHOOSE, PHASE_ABANDON] and next_status == PHASE_CONQUEST:
				if current_ppl[3] == 0:
					how_many_ppl_available += 4

		elif current_ppl[1] == SKELETON:
			if old_status in [PHASE_READY, PHASE_CHOOSE, PHASE_ABANDON, PHASE_CONQUEST, PHASE_CONQ_WITH_DICE, PHASE_ABANDON_AMAZONS] and next_status == PHASE_REDEPLOY:
				if current_ppl[3] == 0:
					how_many_ppl_available += self._limit_added_ppl(current_ppl, current_ppl[3] // 2, MAX_SKELETONS, player_territories)
		return how_many_ppl_available		

	def _switch_to_next(self, player, current_ppl, deterministic):
		if self.game_status[player, 4] != ACTIVE:		# Next turn is for same player
			next_player, next_ppl_id = player, ACTIVE
		else:											# Next turn is for next player
			next_player = (player+1) % NUMBER_PLAYERS
			if self.peoples[next_player, DECLINED_SPIRIT, 1] == -GHOUL:
				next_ppl_id = DECLINED_SPIRIT
			elif self.peoples[next_player, DECLINED, 1] == -GHOUL:
				next_ppl_id = DECLINED
			else:
				next_ppl_id = ACTIVE

			self.game_status[player, 3] += 1
			self.game_status[player, 4] = -1
			self.round_status[player, 4] = PHASE_WAIT

		# Reset stuff depending on people
		if current_ppl[1] == SKELETON:
			current_ppl[3] = 0 # Must be reset
		elif current_ppl[1] == HALFLING:
			pass # Don't reset it
		elif current_ppl[1] == SORCERER:
			current_ppl[3] = 0 # Must be reset
		else:
			pass

		# Reset stuff depending on power
		if current_ppl[2] == WEALTHY:
			pass
		elif current_ppl[2] == BIVOUACKING:
			pass
		elif current_ppl[2] == FORTIFIED:
			# Reset only the "have I used fortress during this turn" part
			remaining_fort, used_fort = _split_pwr_data(current_ppl[4])
			current_ppl[4] = remaining_fort
		elif current_ppl[2] == HEROIC:
			pass
		elif current_ppl[2] == DIPLOMAT:
			pass
		else:
			current_ppl[4] = 0

		# Reset #NETWDT
		self.round_status[player, 3] = 0

		# Switch next ppl to new state
		next_ppl = self.peoples[next_player, next_ppl_id, :]
		self.game_status[next_player, 4] = next_ppl_id
		self.round_status[next_player, 4] = PHASE_READY
		# self._prepare_for_new_status(next_player, next_ppl, PHASE_READY, deterministic) # Numba doesn't allow recursion on jitclass
		self._prepare_for_ready(next_player, next_ppl, deterministic)

	def _compute_and_update_score(self, player):
		current_ppl, _ = self._current_ppl(player)
		self._update_round_status(current_ppl, player)
		score_for_this_turn = 0

		# Iterate on areas and count score
		for area in range(NB_AREAS):
			if self.territories[area, 1] != NOPPL and self.territories[area, 1] in self.peoples[player, :, 1]:
				score_for_this_turn += 1
				# +1 point if: dwarf + mine (even in decline), human + field, wizard + magic, forest + forest,
				#     hill + hill, swamp + swamp, merchant, fortress
				if descr[area][MINE]          and abs(self.territories[area, 1]) == DWARF:
					score_for_this_turn += 1
				if descr[area][0] == FARMLAND and     self.territories[area, 1]  == HUMAN:
					score_for_this_turn += 1
				if descr[area][MAGIC]         and     self.territories[area, 1]  == WIZARD:
					score_for_this_turn += 1
				if descr[area][0] == FORESTT  and     self.territories[area, 2]  == FOREST:
					score_for_this_turn += 1
				if descr[area][0] == HILLT    and     self.territories[area, 2]  == HILL:
					score_for_this_turn += 1
				if descr[area][0] == SWAMPT   and     self.territories[area, 2]  == SWAMP:
					score_for_this_turn += 1
				if                                    self.territories[area, 2]  == MERCHANT:
					score_for_this_turn += 1
				if self.territories[area, 4] > 0 and  self.territories[area, 2]  == FORTIFIED:
					score_for_this_turn += 1

		# Bonus points if: orc (+NETWDT), pillaging (+NETWDT), alchemist (+2), wealthy+1stRound (+7)
		if self.peoples[player, ACTIVE, 1] == ORC:
			score_for_this_turn += self.round_status[player, 3]
		if self.peoples[player, ACTIVE, 2] == PILLAGING:
			score_for_this_turn += self.round_status[player, 3]
		if self.peoples[player, ACTIVE, 2] == ALCHEMIST:
			score_for_this_turn += 2
		if self.peoples[player, ACTIVE, 2] == WEALTHY and self.peoples[player, ACTIVE, 4] > 0:
			score_for_this_turn += self.peoples[player, ACTIVE, 4]
			self.peoples[player, ACTIVE, 4] = 0 # Reset value at this stage

		if score_for_this_turn != self.round_status[player, 6]:
			print(f'Je tombe sur {score_for_this_turn} alors que le calcul itératif donne {self.round_status[player, 6]}')
			# breakpoint()

		backup_score = self.game_status[player, 6]
		self.game_status[player, 6] += score_for_this_turn
		if self.game_status[player, 6] < backup_score:
			print('Overflow protection', backup_score, '+', score_for_this_turn, '=', self.game_status[player, 6])
			self.game_status[player, 6] = 127 # Overflow protection

	def _update_round(self):
		self.game_status[:, 3] += 1

	def _init_deck(self):
		# All people available except NOPPL
		available_people = np.ones(WIZARD+1, dtype=np.int8)
		available_people[NOPPL] = False
		available_power = np.ones(WEALTHY+1, dtype=np.int8)
		available_power[NOPOWER] = False

		# Draw 6 ppl+power randomly
		for i in range(DECK_SIZE):
			chosen_ppl = np.random.choice(np.flatnonzero(available_people))
			chosen_power = np.random.choice(np.flatnonzero(available_power))
			nb_of_ppl = initial_nb_people[chosen_ppl] + initial_nb_power[chosen_power]
			self.visible_deck[i, :] = [nb_of_ppl, chosen_ppl, chosen_power, 0, 0, 0, 0, -1]
			available_people[chosen_ppl], available_power[chosen_power] = False, False

		# Update bitfield
		self.invisible_deck[0:2] = my_packbits(available_people)
		self.invisible_deck[2:5] = my_packbits(available_power)

	def _update_deck_after_chose(self, index, deterministic):
		# Read bitfield
		available_people = my_unpackbits(self.invisible_deck[0:2])
		available_power  = my_unpackbits(self.invisible_deck[2:5])

		# Delete people #item, shift others upwards
		self.visible_deck[index:DECK_SIZE-1, :] = self.visible_deck[index+1:DECK_SIZE, :]
		# Give previous combos 1 coin each
		self.visible_deck[0:index, 6] += 1
		# Draw a new people for last combination
		avail_people_id, avail_power_id = np.flatnonzero(available_people), np.flatnonzero(available_power)
		if USERANDOM:
			if deterministic == 0:
				chosen_ppl = np.random.choice(avail_people_id)
				chosen_power = np.random.choice(avail_power_id)
			else:
				# https://stackoverflow.com/questions/3062746/special-simple-random-number-generator
				# m=avail_people_id.size, c=0, a=2*3*5*7*9*11*13*17+1
				rnd_value = (4594591 * (deterministic+self.invisible_deck[6])) % avail_people_id.size
				chosen_ppl = avail_people_id[rnd_value]
				rnd_value = (4594591 * (deterministic+self.invisible_deck[6])) % avail_power_id.size
				chosen_power = avail_power_id[rnd_value]
			self.invisible_deck[6] += 1
		else:
			chosen_ppl, chosen_power = avail_people_id[2027 % avail_people_id.size], avail_power_id[2027 % avail_power_id.size]
		nb_of_ppl = initial_nb_people[chosen_ppl] + initial_nb_power[chosen_power]
		self.visible_deck[DECK_SIZE-1, :] = [nb_of_ppl, chosen_ppl, chosen_power, 0, 0, 0, 0, -1]
		available_people[chosen_ppl], available_power[chosen_power] = False, False

		# Update back the bitfield
		self.invisible_deck[0:2] = my_packbits(available_people)
		self.invisible_deck[2:5] = my_packbits(available_power)

	def _update_deck_after_decline(self):
		# Note all people as available
		available_people = np.ones(WIZARD+1, dtype=np.int8)
		available_people[NOPPL] = False
		available_power = np.ones(WEALTHY+1, dtype=np.int8)
		available_power[NOPOWER] = False

		# And disable the ones currently used (in deck and in current peoples)
		for ppl in self.visible_deck[:, 1]:
			available_people[ ppl ] = False
		for pwr in self.visible_deck[:, 2]:
			available_power [ pwr ] = False
		for ppl in self.peoples[:, :, 1].flat:
			if ppl != NOPPL:
				available_people[ abs(ppl) ] = False
		for pwr in self.peoples[:, :, 2].flat:
			if pwr != NOPOWER:
				available_power[ abs(pwr) ] = False		

		self.invisible_deck[0:2] = my_packbits(available_people)
		self.invisible_deck[2:5] = my_packbits(available_power)

	def _enough_amazons_to_redeploy(self, player, current_ppl):
		# Check that enough amazon to give back
		if current_ppl[1] == AMAZON:
			how_many_ppl_available = self._ppl_virtually_available(player, current_ppl, PHASE_REDEPLOY)
			if how_many_ppl_available < 0:
				return False
		return True

	def _update_territory_after_win_or_decline(self, current_ppl, player, area):
		# current_ppl just won territory area

		# Compute 3
		if current_ppl[1] == HALFLING and current_ppl[3] > 0:
			self.territories[area, 3] = IMMUNITY
			current_ppl[3] -= 1
		else:
			self.territories[area, 3] = 0
		# Compute 5
		self.territories[area, 5] = self.territories[area, 0] + self.territories[area, 3] + self.territories[area, 4]
		if descr[area][0] == MOUNTAIN:
			self.territories[area, 5] += 1
		if abs(self.territories[area, 1]) == TROLL:
			self.territories[area, 5] += 1
		# Compute 6
		self.territories[area, 6] = 1
		if descr[area][MINE]          and abs(self.territories[area, 1]) == DWARF:
			self.territories[area, 6] += 1
		if descr[area][0] == FARMLAND and     self.territories[area, 1]  == HUMAN:
			self.territories[area, 6] += 1
		if descr[area][MAGIC]         and     self.territories[area, 1]  == WIZARD:
			self.territories[area, 6] += 1
		if descr[area][0] == FORESTT  and     self.territories[area, 2]  == FOREST:
			self.territories[area, 6] += 1
		if descr[area][0] == HILLT    and     self.territories[area, 2]  == HILL:
			self.territories[area, 6] += 1
		if descr[area][0] == SWAMPT   and     self.territories[area, 2]  == SWAMP:
			self.territories[area, 6] += 1
		if                                    self.territories[area, 2]  == MERCHANT:
			self.territories[area, 6] += 1
		if self.territories[area, 4] > 0 and  self.territories[area, 2]  == FORTIFIED:
			self.territories[area, 6] += 1
		# Compute 7
		self.territories[area, 7] = player

	def _update_round_status(self, current_ppl, player):
		current_ppl[6] = 0
		self.round_status[player, 0] = 0
		self.round_status[player, 5] = 0
		self.round_status[player, 6] = 0

		territories_of_player = self._are_occupied_by(current_ppl)
		for area in np.flatnonzero(territories_of_player):
			# Compute peoples[:,:,6]
			current_ppl[6] += self.territories[area, 6]
			# Compute round_status[0]
			self.round_status[player, 0] += self.territories[area, 0]
			# Compute round_status[5]
			# if self.territories[area, 5] > IMMUNITY+10:
			# 	print(f'Overflow protection on IMMUNITY {self.territories[area, 5]}')
			# self.round_status[player, 5] += min(self.territories[area, 5], IMMUNITY+10)
			self.round_status[player, 5] += self.territories[area, 5]
			if self.round_status[player, 5] < 0:
				print(f'Overflow protection on round_status {self.territories[area, 5]} {self.round_status[player, 5]}')
				self.round_status[player, 5] = 127

		if current_ppl[1] >= 0:
			# Bonus points if: orc (+NETWDT), pillaging (+NETWDT), alchemist (+2), wealthy+1stRound (+7)
			if current_ppl[1] == ORC:
				current_ppl[6] += self.round_status[player, 3]
			if current_ppl[2] == PILLAGING:
				current_ppl[6] += self.round_status[player, 3]
			if current_ppl[2] == ALCHEMIST:
				current_ppl[6] += 2
			if current_ppl[2] == WEALTHY and current_ppl[4] > 0:
				current_ppl[6] += current_ppl[4]
				# Reset current_ppl[4] only at the end, in _compute_and_update_score()

		# Compute round_status[6]
		self.round_status[player, 6] = self.peoples[player, :, 6].sum()
