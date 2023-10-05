import numpy as np
from numba import njit
import numba

from SmallworldConstants import *
from SmallworldDisplay import print_board, print_valids

############################## BOARD DESCRIPTION ##############################

# Board is described by a NB_AREAS+4*NUMBER_PLAYERS+7 x 5 array
# (38x5 for 2 players for instance, 38 = 23 + 4*2 + 7)
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
#####  0-22  self.territories        0-4     Same, in territory i
#####   23   self.peoples (3D)     0   0-2   Generic info (nb, type, power) for declined+spirit people in player 0's hand
#####                              0    3    Various info related to people capacity for declined+spirit people in player 0's hand
#####                              0    4    Various info related to special power for declined+spirit people in player 0's hand
#####   24                         1   0-4   Same for declined (non-spirit) people in player 0's hand
#####   25                         2   0-4   Same for active people in player 0's hand
#####  26-28                       0-2 0-4   Same info for peoples in player 1's hand
#####  29-34 self.visible_deck       0-2     Generic info (nb, type, power) for each combo on deck
#####                                 3      Number of victory points to win with such combo
#####                                 4      - empty -
#####  35-36 self.status              0      Score of player i
#####                                 1      Round number (same for all players)
#####                                 2      #NETWDT = number of Non-Empty Territories Won During Turn
#####                                 3      Id of people (0 or 1 or 2) current playing, -1 else
#####                                 4      Status of people (from PHASE_READ to PHASE_WAIT)
#####   37   self.invisible_deck(1D) 0-1     Bitfield stating if people #i (0-14) is already in deck or used
#####                                2-4     Bitfield stating if power #i (0-20) is already in deck or used
# Indexes above are assuming 2 players, you can have more details in copy_state().
#
# How is used self.peoples[:,:,3] depending on people:
#  Amazon  : number of people under "loan", 4 during attack and should be 0 at end of turn
#  Skeleton: 1 if their power was applied this turn
#  Halfling: number of used holes-in-ground
#  Sorcerer: bitfield of which player has been sorcerized during this turn

# @njit(cache=True, fastmath=True, nogil=True)
def observation_size():
	return (NB_AREAS + 4*NUMBER_PLAYERS + DECK_SIZE+1, 5)

# @njit(cache=True, fastmath=True, nogil=True)
def action_size():
	return 10


mask = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint16)

# @njit(cache=True, fastmath=True, nogil=True)
def my_packbits(array):
	padded_array = np.array(array) if len(array)%8 == 0 else np.pad(array, (0, 8-len(array)%8))
	result = np.dot(padded_array.reshape((-1, 8)).astype(np.uint16), mask)
	return result.astype(np.int8)

# @njit(cache=True, fastmath=True, nogil=True)
def my_unpackbits(values):
	result = np.zeros((len(values), 8), dtype=np.uint8)
	for i, v in enumerate(values):
		result[i, :] = (np.bitwise_and(v, mask) != 0)
	return result.flatten()

# @njit(cache=True, fastmath=True, nogil=True)
def my_random_choice(prob):
	result = np.searchsorted(np.cumsum(prob), np.random.random(), side="right")
	return result


spec = [
	('state'         , numba.int8[:,:]),
	('territories'   , numba.int8[:,:]),
	('peoples'       , numba.int8[:,:]),
	('visible_deck'  , numba.int8[:,:]),
	('status'        , numba.int8[:,:]),
	('invisible_deck', numba.int8[:,:]),
]
# @numba.experimental.jitclass(spec)
class Board():
	def __init__(self, num_players):
		self.state = np.zeros(observation_size(), dtype=np.int8)
		self.init_game()

	def get_score(self, player):
		return self.scores[player, 0]

	def init_game(self):
		self.copy_state(np.zeros(observation_size(), dtype=np.int8), copy_or_not=False)
		for i in range(NB_AREAS):
			if descr[i][3]:
				nb_lt = initial_nb_people[LOST_TRIBE]
				self.territories[i,:] = [nb_lt, LOST_TRIBE, NOPOWER, 0, 0]
			else:
				self.territories[i,:] = [0    , NOPPL     , NOPOWER, 0, 0]

		# Init deck of people and draw for P0 and P1
		self._init_deck()
		for player in range(NUMBER_PLAYERS):
			self._do_choose_ppl(player, 0)
			self.status[player, 0] = 0 # Reset money
			self.status[player, 3:] = [-1, PHASE_WAIT]
		self.status[0, 3:] = [ACTIVE, PHASE_READY]

		# First round is round #1
		self._update_round()
	
	def copy_state(self, state, copy_or_not):
		if self.state is state and not copy_or_not:
			return
		self.state = state.copy() if copy_or_not else state

		self.territories    = self.state[0                                  :NB_AREAS                             ,:]
		self.peoples        = self.state[NB_AREAS                           :NB_AREAS+3*NUMBER_PLAYERS            ,:]
		self.visible_deck   = self.state[NB_AREAS+3*NUMBER_PLAYERS          :NB_AREAS+3*NUMBER_PLAYERS+DECK_SIZE  ,:]
		self.status         = self.state[NB_AREAS+3*NUMBER_PLAYERS+DECK_SIZE:NB_AREAS+4*NUMBER_PLAYERS+DECK_SIZE  ,:]
		self.invisible_deck = self.state[NB_AREAS+4*NUMBER_PLAYERS+DECK_SIZE                                      ,:]
		self.peoples = self.peoples.view().reshape(NUMBER_PLAYERS, 3, -1)

	def valid_moves(self, player):
		result = np.zeros(10, dtype=np.bool_)
		return result

	def make_move(self, move, player, deterministic):
		return 1-player

	def get_state(self):
		return self.state

	def get_round(self):
		return self.status[0,1]

	def check_end_game(self, next_player):
		if self.get_round() <= 10:
			return np.array([0, 0], dtype=np.float32) # No winner yet

		# Game is ended
		scores = self.status[:, 0]
		best_score = scores.max()
		return np.where(scores == best_score, 1, -1).astype(np.float32)

	def swap_players(self, nb_swaps):
		return

	def get_symmetries(self, policy, valids):
		# Always called on canonical board, meaning player = 0
		# In all symmetries, no need to update the "optim" vectors as they are not used by NN
		symmetries = [(self.state.copy(), policy.copy(), valids.copy())]
		state_backup, policy_backup, valids_backup = symmetries[0]

		return symmetries

	###########################################################################

	def _valids_attack(self, player):
		valids = np.zeros(NB_AREAS, dtype=np.bool_)
		current_ppl, current_id = self._current_ppl(player)

		# Check that player has a valid ppl:
		if current_ppl[1] == NOPPL:
			return valids

		# Attack permitted when it's the time
		if self.status[player, 4] not in [PHASE_READY, PHASE_CHOOSE, PHASE_ABANDON, PHASE_CONQUEST]:
			return valids

		territories_of_player = self._are_occupied_by(current_ppl)
		how_many_ppl_available = self._ppl_virtually_available(player, current_ppl, PHASE_CONQUEST, territories_of_player)
		# Forbid to continue if 0 ppl left in hand
		if how_many_ppl_available <= 0:
			return valids

		for area in range(NB_AREAS):
			valids[area] = self._valid_attack_area(player, area, current_ppl, how_many_ppl_available, territories_of_player)

		return valids

	def _valid_attack_area(self, player, area, current_ppl, how_many_ppl_available, territories_of_player):
		# No attack on water
		if descr[area][0] == WATER:
			return False

		# No attack on current people
		if self._is_occupied_by(area, current_ppl):
			return False

		# Check no immunity
		if self.territories[area, 3] >= IMMUNE_CONQUEST or self.territories[area, 4] >= IMMUNE_CONQUEST:
			return False

		# Check that player has a chance to win	
		minimum_ppl_for_attack = self._minimum_ppl_for_attack(area, current_ppl)
		if how_many_ppl_available + MAX_DICE < minimum_ppl_for_attack:
			return False

		# Check that territory is close to another owned territory or is on the edge
		if np.count_nonzero(territories_of_player) == 0:
			area_is_on_edge = descr[area][2]
			if current_ppl[1] != HALFLING and not area_is_on_edge:
				return False
		else:
			neighbor_areas = connexity_matrix[area]
			if not np.any(np.logical_and(neighbor_areas, territories_of_player)):
				return False

		return True

	def _do_attack(self, player, area):
		current_ppl, current_id = self._current_ppl(player)

		# If 1st action of the turn, prepare people
		self._prepare_for_new_status(player, current_ppl, PHASE_CONQUEST)

		nb_ppl_of_player, minimum_ppl_for_attack = current_ppl[0],self._minimum_ppl_for_attack(area, current_ppl)

		# Use dice if people are needed
		use_dice = (nb_ppl_of_player < minimum_ppl_for_attack)
		if use_dice:
			dice = DICE_VALUES[3]
			if nb_ppl_of_player + dice < minimum_ppl_for_attack:
				print(f'  Using dice, random value is {dice} but fails')
				self.status[player, 4] = PHASE_CONQ_WITH_DICE
				return
			print(f'  Using dice, random value is {dice} and succeed')
			nb_attacking_ppl = nb_ppl_of_player
		else:
			nb_attacking_ppl = minimum_ppl_for_attack

		# Attack is successful

		# Update loser and winner
		self._switch_territory_from_loser_to_winner(area, player, current_ppl, nb_attacking_ppl)

		# Update winner's status
		self.status[player, 4] = PHASE_CONQ_WITH_DICE if use_dice else PHASE_CONQUEST

	def _valids_redeploy(self, player):
		valids = np.zeros(NB_AREAS + MAX_REDEPLOY, dtype=np.bool_)
		current_ppl, current_id = self._current_ppl(player)

		# Check that player has a valid ppl:
		if current_ppl[1] == NOPPL:
			return valids

		# Check that it is time
		if self.status[player, 4] not in [PHASE_READY, PHASE_ABANDON, PHASE_CONQUEST, PHASE_CONQ_WITH_DICE, PHASE_REDEPLOY]:
			return valids

		# Check there is at least one active territory
		territories_of_player = self._are_occupied_by(current_ppl)
		nb_territories = np.count_nonzero(territories_of_player)
		if nb_territories == 0:
			valids[0] = True # If no other option, then allow to skip redeploy
			return valids

		# Check that player has still some ppl to deploy
		how_many_ppl_available = self._ppl_virtually_available(player, current_ppl, PHASE_REDEPLOY, territories_of_player)
		if how_many_ppl_available == 0:
			valids[0] = True # If no other option, then allow to skip redeploy
			return valids
		elif how_many_ppl_available < 0:
			return valids    # Redeploy not allowed, need to abandon instead

		for ppl_to_deploy in range(1, MAX_REDEPLOY):
			valids[ppl_to_deploy] = (how_many_ppl_available >= ppl_to_deploy * nb_territories)
		for area in range(NB_AREAS):
			valids[MAX_REDEPLOY + area] = territories_of_player[area]

		if not valids.any():
			valids[0] = True # If no other option, then allow to skip redeploy

		return valids

	def _do_redeploy(self, player, param):
		current_ppl, current_id = self._current_ppl(player)

		if param == 0: # Special case, skip redeploy
			self._prepare_for_new_status(player, current_ppl, PHASE_REDEPLOY)
			self._prepare_for_new_status(player, current_ppl, PHASE_WAIT)
			# Status already changed by previous function
			return

		self._prepare_for_new_status(player, current_ppl, PHASE_REDEPLOY)
		self.status[player, 4] = PHASE_REDEPLOY

		if param < MAX_REDEPLOY:
			# Deploy X ppl on all active areas
			how_many_to_deploy = param

			territories_of_player = self._are_occupied_by(current_ppl)
			current_ppl[0] -= how_many_to_deploy * np.count_nonzero(territories_of_player)
			assert current_ppl[0] >= 0
			self.territories[:, 0] += how_many_to_deploy*territories_of_player
		else:
			# Deploy 1 ppl on 1 area
			area = param - MAX_REDEPLOY

			current_ppl[0]            -= 1
			self.territories[area, 0] += 1

		# Trigger end of turn if no more to redeploy
		if current_ppl[0] == 0:
			self._prepare_for_new_status(player, current_ppl, PHASE_WAIT)
			# Status already changed by previous function

	def _valid_decline(self, player):
		# Going to decline permitted only for active_ppl
		if self.status[player, 3] != ACTIVE or self.peoples[player, ACTIVE, 1] == NOPPL:
			return False
		# Going to decline permitted only on 1st move
		if self.status[player, 4] != PHASE_READY:
			return False
		return True

	def _do_decline(self, player):
		current_ppl = self.peoples[player, ACTIVE, :]
		# Remove previous declined ppl from the board
		for area in range(NB_AREAS):
			if self._is_occupied_by(area, self.peoples[player, DECLINED, :]):
				self.territories[area] = [0, NOPPL, NOPOWER, 0, 0]

		# Move ppl to decline and keep only 1 ppl per territory
		self.peoples[player, DECLINED, :] = 0
		if current_ppl[1] == GHOUL:
			self.peoples[player, DECLINED, :3] = current_ppl[:3]
		else:
			self._gather_current_ppl_but_one(current_ppl)
			self.peoples[player, DECLINED, 1] = current_ppl[1]
		current_ppl[:] = [0, NOPPL, NOPOWER, 0, 0]
		
		# Flip back ppl tokens on the board and remove defense
		for area in range(NB_AREAS):
			if self.territories[area, 1] == self.peoples[player, DECLINED, 1]:
				self.territories[area, 1] = -self.peoples[player, DECLINED, 1]
				self.territories[area, 2:] = 0
		self.peoples[player, DECLINED, 1:3] = -self.peoples[player, DECLINED, 1:3]
		
		self._prepare_for_new_status(player, current_ppl, PHASE_WAIT)
		self.status[player, 4] = PHASE_WAIT

	def _valids_choose_ppl(self, player):
		valids = np.zeros(DECK_SIZE, dtype=np.bool_)

		# Check that it is time
		if self.status[player, 4] != PHASE_READY:
			return valids
		# Check that player hasn't a player yet
		if self.peoples[player, ACTIVE, 1] != NOPPL:
			return valids

		for index in range(DECK_SIZE):
			# Check that index is valid and player can pay
		 	valids[index] = (self.visible_deck[index, 1] != NOPPL) and (self.status[player, 0] >= index)

		return valids

	def _do_choose_ppl(self, player, index):
		current_ppl = self.peoples[player, ACTIVE, :]

		current_ppl[:]  = 0
		current_ppl[:3] = self.visible_deck[index, :3]
		current_ppl[3]  = initial_tokens[self.visible_deck[index, 1]]

		# Earn money but also pay what's needed
		print(f'Earn {self.visible_deck[index, 3]} but pay {index}')
		self.status[player, 0] += self.visible_deck[index, 3] - index

		self._prepare_for_new_status(player, current_ppl, PHASE_CHOOSE)
		self.status[player, 4] = PHASE_CHOOSE
		self._update_deck_after_chose(index)

	def _valids_abandon(self, player):
		valids = np.zeros(NB_AREAS, dtype=np.bool_)
		current_ppl, current_id = self._current_ppl(player)

		if self.status[player, 4] not in [PHASE_READY, PHASE_ABANDON]:
			if current_ppl[1] == AMAZON and self.status[player, 4] in [PHASE_CONQUEST, PHASE_CONQ_WITH_DICE] and self._ppl_virtually_available(player, current_ppl, PHASE_REDEPLOY) < 0:
				pass # exception if Amazons can't redeploy
			else:
				return valids
		# Cant abandon if player doesn't have any active people
		if current_ppl[1] == NOPPL:
			return valids

		for area in range(NB_AREAS):
			valids[area] = self._is_occupied_by(area, current_ppl)

		return valids

	def _do_abandon(self, player, area):
		current_ppl, current_id = self._current_ppl(player)
		self._leave_area(area)
		if self.status[player, 4] in [PHASE_CONQUEST, PHASE_CONQ_WITH_DICE]:
			pass # exception if Amazons can't redeploy, don't change status
		else:
			self._prepare_for_new_status(player, current_ppl, PHASE_ABANDON)
			self.status[player, 4] = PHASE_ABANDON

	def _valids_special_action(self, player):
		valids = np.zeros(NB_AREAS, dtype=np.bool_)
		current_ppl, current_id = self._current_ppl(player)

		if current_ppl[1] == SORCERER:
			# Attack permitted when it's the time
			if self.status[player, 4] not in [PHASE_READY, PHASE_CHOOSE, PHASE_ABANDON, PHASE_CONQUEST]:
				return valids

			# Limited number of sorcerers in the box
			territories_of_player = self._are_occupied_by(current_ppl)
			total_number = self._total_number_of_ppl(current_ppl, territories_of_player)
			if total_number + 1 > MAX_SORCERERS:
				return valids
			
			for area in range(NB_AREAS):
				valids[area] = self._valid_special_action_area(player, area, current_ppl, territories_of_player)

		return valids

	def _valid_special_action_area(self, player, area, current_ppl, territories_of_player):
		if current_ppl[1] == SORCERER:
			# No attack on water
			if descr[area][0] == WATER:
				return False
			# People on this area is alone and active
			if self.territories[area, 0] != 1 or self.territories[area, 1] <= 0:
				return False
			# No attack on current people
			if self._is_occupied_by(area, current_ppl):
				return False
			# Check no full immunity
			if self.territories[area, 3] >= FULL_IMMUNITY or self.territories[area, 4] >= FULL_IMMUNITY:
				return False
			# Check that territory is close to another owned territory or is on the edge
			neighbor_areas = connexity_matrix[area]
			if not np.any(np.logical_and(neighbor_areas, territories_of_player)):
				return False
			# Check that opponent had not been already 'sorcerized' during this turn
			_, loser = self._ppl_owner_of(area)
			if current_ppl[3] & 2**loser:
				return False

			return True

		else:
			return False

	def _do_special_action(self, player, area):
		current_ppl, current_id = self._current_ppl(player)

		if current_ppl[1] == SORCERER:
			loser_ppl, loser = self._ppl_owner_of(area)
			# Prepare people if 1st action of the turn
			self._prepare_for_new_status(player, current_ppl, PHASE_CONQUEST)
			# Replace
			self.territories[area,:] = [1, SORCERER, current_ppl[2], 0, 0]
			# Note that loser have been 'sorcerized'
			current_ppl[3] += 2**loser
			# Update winner's status and #NETWDT
			self.status[player, 4] = PHASE_CONQUEST
			current_ppl[2] += 1

		else:
			raise Exception('Should not happen')

	###########################################################################

	def _current_ppl(self, player):
		current_id = self.status[player, 3]
		if current_id < 0:
			raise Exception(f'No ppl to play for P{player}')
		return self.peoples[player, current_id, :], current_id

	def _ppl_owner_of(self, area):
		if self.territories[area, 1] == NOPPL or self.territories[area, 1] == LOST_TRIBE:
			return None, -1
		result = np.argwhere(self.peoples[:,:,1] == self.territories[area, 1])
		if result.shape[0] != 1:
			breakpoint()
			raise Exception(f'Could not find which ppl this area belongs ({area=} {self.territories[area, 1]=} {result=})')
		return self.peoples[result[0][0], result[0][1], :], result[0][0]

	def _is_occupied_by(self, area, current_ppl):
		return (self.territories[area,1] == current_ppl[1])

	def _are_occupied_by(self, current_ppl):
		return (self.territories[:,1] == current_ppl[1])

	def _is_area_border_of(self, area, terrain):
		neighbor_areas = connexity_matrix[area]
		areas_with_terrain = (np.array(descr)[:,0] == terrain)
		result = np.any(np.logical_and(neighbor_areas, areas_with_terrain)).item()
		return result

	def _minimum_ppl_for_attack(self, area, current_ppl):
		minimum_ppl_for_attack = self.territories[area,0] + self.territories[area, 3] + self.territories[area, 4] + 2

		# Malus if: mountain, troll (even in decline)
		if descr[area][0] == MOUNTAIN:
			minimum_ppl_for_attack += 1
		if abs(self.territories[area, 1]) == TROLL:
			minimum_ppl_for_attack += 1

		# Bonus if: triton + at_edge, giant + border of mountain
		if current_ppl[1] == TRITON and descr[area][2]:
			minimum_ppl_for_attack -= 1
		if current_ppl[1] == GIANT  and self._is_area_border_of(area, MOUNTAIN):
			minimum_ppl_for_attack -= 1

		return max(minimum_ppl_for_attack, 1)

	def _leave_area(self, area):
		# Give back ppl to owner
		self._ppl_owner_of(area)[0][0] += self.territories[area, 0]

		# Make the area empty
		self.territories[area,:] = 0

	def _switch_territory_from_loser_to_winner(self, area, player, winner_ppl, nb_attacking_ppl):
		nb_initial_ppl = self.territories[area, 0]

		# Give back people to the loser (if any)
		loser_ppl, _ = self._ppl_owner_of(area)
		if loser_ppl is not None:
			assert nb_initial_ppl > 0
			# Lose 1 ppl unless you are elf and active
			nb_ppl_to_lose = 1 if self.territories[area,1] != ELF else 0
			loser_ppl[0] += self.territories[area,0] - nb_ppl_to_lose

		# Install people from the winner
		self.territories[area,0] = nb_attacking_ppl
		self.territories[area,1:3] = winner_ppl[1:3]
		self.territories[area,3:] = 0
		winner_ppl[0] -= nb_attacking_ppl
		assert winner_ppl[0] >= 0
		# Add specific tokens
		if winner_ppl[1] == HALFLING and winner_ppl[3] > 0:
			self.territories[area, 3] = FULL_IMMUNITY
			winner_ppl[3] -= 1

		# Update #NETWDT
		if nb_initial_ppl > 0:
			self.status[player, 2] += 1

	def _total_number_of_ppl(self, current_ppl, player_territories=None):
		if player_territories is None:
			player_territories = self._are_occupied_by(current_ppl)

		how_many_on_board = np.dot(self.territories[:,0], player_territories)
		how_many_in_hand  = current_ppl[0]
		return how_many_on_board + how_many_in_hand

	def _limit_added_ppl(self, current_ppl, addition, maximum, player_territories=None):
		total_number = self._total_number_of_ppl(current_ppl, player_territories)
		return min(addition, maximum - total_number)

	def _gather_current_ppl_but_one(self, current_ppl):
		# Gather all active people in player's hand, leaving only 1 on each territory
		# print(f'Prepare / redeploy P{player}:', end='')

		for area in range(NB_AREAS):
			if self._is_occupied_by(area, current_ppl):
				nb_ppl_to_gather = max(self.territories[area,0] - 1, 0)
				if nb_ppl_to_gather > 0:
					self.territories[area,0] -= nb_ppl_to_gather
					current_ppl[0]           += nb_ppl_to_gather
					# print(f' {nb_ppl_to_gather}ppl on area{area}', end='')

	def _prepare_for_new_status(self, player, current_ppl, next_status):
		old_status = self.status[player, 4]

		if old_status in [PHASE_READY] and next_status in [PHASE_ABANDON, PHASE_CONQUEST, PHASE_CONQ_WITH_DICE]:
			self._gather_current_ppl_but_one(current_ppl)
		elif old_status in [PHASE_READY, PHASE_CONQUEST, PHASE_CONQ_WITH_DICE] and next_status == PHASE_REDEPLOY:
			self._gather_current_ppl_but_one(current_ppl)
		elif next_status == PHASE_WAIT:
			if self.status[player, 3] == ACTIVE:
				self._compute_and_update_score(player)
			self._switch_to_next(player, current_ppl)

		if current_ppl[1] == AMAZON:
			self._switch_status_amazon(player, current_ppl, old_status, next_status)
		elif current_ppl[1] == SKELETON:
			self._switch_status_skeleton(player, current_ppl, old_status, next_status)	

	def _switch_status_amazon(self, player, current_ppl, old_status, next_status):
		if old_status in [PHASE_CONQUEST, PHASE_CONQ_WITH_DICE] and next_status == PHASE_REDEPLOY:
			if current_ppl[3] != 0:
				print(f'Remove bonus Amazons {current_ppl[0]} --> {current_ppl[0]-current_ppl[3]}')
				current_ppl[0] -= current_ppl[3]
				current_ppl[3]  = 0
			else:
				print(f'Bonus Amazone not used ? {current_ppl[3]}')
				breakpoint()

		elif old_status in [PHASE_READY, PHASE_CHOOSE, PHASE_ABANDON] and next_status == PHASE_CONQUEST:
			if current_ppl[3] == 0:
				print(f'Add bonus Amazons {current_ppl[0]} --> {current_ppl[0]+4}')
				current_ppl[0] += 4
				current_ppl[3]  = 4
			else:
				print(f'Bonus Amazone already used ? {current_ppl[3]}')
				breakpoint()

	def _switch_status_skeleton(self, player, current_ppl, old_status, next_status):
		if old_status in [PHASE_CONQUEST, PHASE_CONQ_WITH_DICE] and next_status == PHASE_REDEPLOY:
			if current_ppl[3] == 0:
				current_ppl[0] += self._limit_added_ppl(current_ppl, current_ppl[3] // 2, MAX_SKELETONS)
				current_ppl[3] = 1 # We write that we gave additional ppl already
			else:
				print(f'Skeleton power already used ? {current_ppl[3]}')
				breakpoint()

	def _ppl_virtually_available(self, player, current_ppl, next_status, player_territories=None):
		if player_territories is None:
			player_territories = self._are_occupied_by(current_ppl)

		old_status = self.status[player, 4]
		how_many_ppl_available = current_ppl[0]
		if old_status in [PHASE_READY] and next_status in [PHASE_ABANDON, PHASE_CONQUEST, PHASE_CONQ_WITH_DICE]:
			# Simulate redeploy: add people on the boards, except 1 per territory
			how_many_ppl_available += np.dot(np.maximum(self.territories[:,0]-1,0), player_territories)
		elif old_status in [PHASE_READY, PHASE_CONQUEST, PHASE_CONQ_WITH_DICE] and next_status == PHASE_REDEPLOY:
			# Simulate redeploy: add people on the boards, except 1 per territory
			how_many_ppl_available += np.dot(np.maximum(self.territories[:,0]-1,0), player_territories)

		# Additional people depending on people and game status
		if current_ppl[1] == AMAZON:
			if old_status in [PHASE_CONQUEST, PHASE_CONQ_WITH_DICE] and next_status == PHASE_REDEPLOY:
				if current_ppl[3] != 0:
					how_many_ppl_available -= current_ppl[3]
				else:
					print(f'Bonus Amazone not used ? {current_ppl[3]}')
					breakpoint()

			elif old_status in [PHASE_READY, PHASE_CHOOSE, PHASE_ABANDON] and next_status == PHASE_CONQUEST:
				if current_ppl[3] == 0:
					how_many_ppl_available += 4
				else:
					print(f'Bonus Amazone already used ? {current_ppl[3]}')
					breakpoint()

		elif current_ppl[1] == SKELETON:
			if old_status in [PHASE_CONQUEST, PHASE_CONQ_WITH_DICE] and next_status == PHASE_REDEPLOY:
				if current_ppl[3] == 0:
					how_many_ppl_available += self._limit_added_ppl(current_ppl, current_ppl[3] // 2, MAX_SKELETONS, player_territories)
				else:
					print(f'Skeleton power already used ? {current_ppl[3]}')
					breakpoint()
		return how_many_ppl_available		

	def _switch_to_next(self, player, current_ppl):
		if self.status[player, 3] == ACTIVE:				# Next turn is for next player
			next_player = (player+1) % NUMBER_PLAYERS
			if next_player == 0:
				self._update_round()

			if self.peoples[next_player, DECLINED_SPIRIT, 1] == -GHOUL:
				next_ppl_id = DECLINED_SPIRIT
			elif self.peoples[next_player, DECLINED, 1] == -GHOUL:
				next_ppl_id = DECLINED
			else:
				next_ppl_id = ACTIVE
			next_ppl = self.peoples[next_player, next_ppl_id, :]

			self.status[player, 3:5] = [-1, PHASE_WAIT]
			assert self.status[next_player, 3] == -1
			assert self.status[next_player, 4] == PHASE_WAIT
			self.status[next_player, 3], self.status[next_player, 4] = next_ppl_id, PHASE_READY
		else:  												# Next turn is for same player
			print('Same player to play with its active ppl now')
			next_player = player
			next_ppl = self.peoples[player, ACTIVE, :]
			self.status[player, 3:5] = [ACTIVE, PHASE_READY]

		# Reset stuff depending on people
		if current_ppl[1] == SKELETON:
			if current_ppl[3] != 1:
				print('** skeleton power hasnt been applied during this turn')
				breakpoint()
			current_ppl[3] = 0 # Must be reset
		elif current_ppl[1] == HALFLING:
			pass # Don't reset it
		elif current_ppl[1] == SORCERER:
			current_ppl[3] = 0 # Must be reset
		else:
			if current_ppl[3] != 0:
				print(f'People info not null with people={current_ppl[1]}')
				breakpoint()

		# Reset stuff depending on power
		current_ppl[4] = 0

		# Reset #NETWDT
		self.status[player, 2] = 0

	def _compute_and_update_score(self, player):
		score_for_this_turn = 0

		# Iterate on areas and count score
		for area in range(NB_AREAS):
			if self.territories[area, 1] != NOPPL and self.territories[area, 1] in self.peoples[player, :, 1]:
				score_for_this_turn += 1
				# +1 point if: dwarf + mine (even in decline), human + field, wizard + magic
				if descr[area][1] == MINE     and abs(self.territories[area, 1]) == DWARF:
					score_for_this_turn += 1
				if descr[area][0] == FARMLAND and     self.territories[area, 1]  == HUMAN:
					score_for_this_turn += 1
				if descr[area][1] == MAGIC    and     self.territories[area, 1]  == WIZARD:
					score_for_this_turn += 1

		# +1 point if: orc + NETWDT
		if self.peoples[player, ACTIVE, 1] == ORC:
			score_for_this_turn += self.status[player, 2]

		self.status[player, 0] += score_for_this_turn

		# Reset NETWDT
		self.status[player, 2] = 0

	def _update_round(self):
		self.status[:, 1] += 1

	def _init_deck(self):
		# All people available except NOPPL and LOST_TRIBE
		available_people = np.ones(LOST_TRIBE, dtype=np.int8)
		available_people[NOPPL] = False
		available_power = np.ones(WEALTHY+1, dtype=np.int8)
		available_power[NOPOWER] = False

		# Draw 6 ppl+power randomly
		for i in range(DECK_SIZE):
			chosen_ppl = my_random_choice(available_people / available_people.sum())
			# chosen_ppl = [SKELETON, AMAZON, SORCERER, GHOUL, TROLL, GIANT, TRITON, HUMAN, WIZARD, DWARF, ELF][i]
			chosen_power = my_random_choice(available_power / available_power.sum())
			self.visible_deck[i, :] = [initial_nb_people[chosen_ppl], chosen_ppl, chosen_power, 0, 0]
			available_people[chosen_ppl], available_power[chosen_power] = False, False

		# Update bitfield
		self.invisible_deck[:2] = my_packbits(available_people)
		self.invisible_deck[2:] = my_packbits(available_power)

	def _update_deck_after_chose(self, index):
		# Read bitfield
		available_people = my_unpackbits(self.invisible_deck[:2])
		available_power  = my_unpackbits(self.invisible_deck[2:])

		# Delete people #item, shift others upwards and give them 1 coin each
		self.visible_deck[index:DECK_SIZE-1, :] = self.visible_deck[index+1:DECK_SIZE, :]
		self.visible_deck[index:DECK_SIZE-1, 3] += 1
		# Draw a new people for last combination
		chosen_ppl   = my_random_choice(available_people / available_people.sum())
		chosen_power = my_random_choice(available_power / available_power.sum())
		self.visible_deck[DECK_SIZE-1, :] = [initial_nb_people[chosen_ppl], chosen_ppl, chosen_power, 0, 0]
		available_people[chosen_ppl], available_power[chosen_power] = False, False

		# Update back the bitfield
		self.invisible_deck[:2] = my_packbits(available_people)
		self.invisible_deck[2:] = my_packbits(available_power)

###############################################################################
import random

b = Board(NUMBER_PLAYERS)
print_board(b)
print()

def play_one_turn():
	p = (b.status[:, 3] >= 0).nonzero()[0].item()
	print(f'Player is now P{p}')

	while b.status[p, 3] >= 0:
		# breakpoint()
		valids_attack    = b._valids_attack(player=p)
		valids_special   = b._valids_special_action(player=p)
		valids_abandon   = b._valids_abandon(player=p)
		valids_redeploy  = b._valids_redeploy(player=p)
		valids_choose    = b._valids_choose_ppl(player=p)
		valid_decline    = b._valid_decline(player=p)
		print_valids(p, valids_attack, valids_special, valids_abandon, valids_redeploy, valids_choose, valid_decline)

		if any(valids_attack) or any(valids_special) or any(valids_abandon) or valid_decline:
			values = np.concatenate((valids_attack.nonzero()[0], valids_special.nonzero()[0] + NB_AREAS, valids_abandon.nonzero()[0] + 2*NB_AREAS, ([3*NB_AREAS] if valid_decline else [])), axis=None)
			dice = np.random.choice(values.astype(np.int64))
			if dice < NB_AREAS:
				area = dice
				print(f'Attacking area {area}')
				b._do_attack(player=p, area=area)
			elif dice < 2*NB_AREAS:
				area = dice - NB_AREAS
				print(f'Special move on area {area}')
				b._do_special_action(player=p, area=area)
			elif dice < 3*NB_AREAS:
				area = dice - 2*NB_AREAS
				print(f'Abandonning area {area}')
				b._do_abandon(player=p, area=area)
			else:
				print(f'*** Decline current ppl')
				b._do_decline(player=p)

		elif any(valids_choose):
			chosen_ppl = np.random.choice(valids_choose.nonzero()[0].astype(np.int64))
			print(f'Choose ppl #{chosen_ppl}')
			b._do_choose_ppl(player=p, index=chosen_ppl)	

		elif any(valids_redeploy):
			valids_on_each = valids_redeploy[:MAX_REDEPLOY]

			# Chose a "redeploy on each" action if possible
			if valids_on_each.any():
				param = valids_on_each.nonzero()[0].max()
				print(f'Redeploy {param}ppl on each area')
			else:
				area = np.random.choice(valids_redeploy[MAX_REDEPLOY:].nonzero()[0].astype(np.int64))
				param = area + MAX_REDEPLOY
				print(f'Redeploy on area {area}')
			
			b._do_redeploy(player=p, param=param)

		else:
			breakpoint()
			break 

		print('-'*40)
		print_board(b)
		print('-'*40)

	return 1-p

p = 0
while not b.check_end_game(p).any():
	p = play_one_turn()

print(f'The end: {b.check_end_game(p)}')
