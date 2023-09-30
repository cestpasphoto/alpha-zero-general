import numpy as np
from numba import njit
import numba

from SmallworldConstants import *
from SmallworldDisplay import print_board, print_valids

############################## BOARD DESCRIPTION ##############################
# state = [

       # self.territories
       # defense = 1 or IMMUNE_CONQUEST or FULL_IMMUNITY
#nbPPL TypePPL  defense Player  ?
#  [0,   NOPPL  , 0     , -1 ,  0], # 0
#  [0,   NOPPL  , 0     , -1 ,  0], # 1
#  [2,   -PRIMIT, 0     , -1 ,  0], # 2
#  [0,   NOPPL  , 0     , -1 ,  0], # 3
#  [2,   -PRIMIT, 0     , -1 ,  0], # 4

		# self.active_ppl
		# #NETWDT = number of non-empty territories won during turn
#nbPPL TypePPL   status    #NETWDT #tokens
#  [7, OGRE   , TO_CONQUEST, 0,     0 ], # Active ppl in player 0's hand
#  [9, DWARF  , TO_CONQUEST, 0,     0 ], # Active ppl in player 1's hand

		# self.decline_ppl
#  ?    TypePPL   status      ?
#  [0,   -HUMAN , 0  ,   0,  0 ], # Declined ppl in player 0's hand
#  [0,   NOPPL  , 0  ,   0,  0 ], # Declined ppl in player 1's hand

		# self.scores
# score  turn    ?      ?   ?
#  [10,   0    , 0  ,   0,  0 ], # Score of player 0
#  [9 ,   0    , 0  ,   0,  0 ], # Score of player 1

		# self.people_deck
# nbPPL TypePPL   cost   ?   ?
# [ 5,   HUMAN,   1  ,   0,  0 ]
# [ 5,   TYPE2,   2  ,   0,  0 ]
# [ 5,   TYPE3,   3  ,   0,  0 ]
# [ 5,   TYPE4,   4  ,   0,  0 ]
# [ 5,   TYPE5,   5  ,   0,  0 ]
# [ 5,   TYPE6,   6  ,   0,  0 ]
# [ bitfieldPPL, bitfieldPPL, 0, 0 ] 14 ppl (2octets) + 20 pouvoirs (3octets)

# ]


# @njit(cache=True, fastmath=True, nogil=True)
def observation_size():
	return (NB_AREAS + 3*NUMBER_PLAYERS + DECK_SIZE+1, 5)

# @njit(cache=True, fastmath=True, nogil=True)
def action_size():
	return 10


mask = np.array([32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint16)

# @njit(cache=True, fastmath=True, nogil=True)
def my_packbits(array):
	product = np.multiply(array.astype(np.uint16), mask[:len(array)])
	return product.sum()

# @njit(cache=True, fastmath=True, nogil=True)
def my_unpackbits(value):
	return (np.bitwise_and(value, mask) != 0).astype(np.uint8)

# @njit(cache=True, fastmath=True, nogil=True)
def my_random_choice(prob):
	result = np.searchsorted(np.cumsum(prob), np.random.random(), side="right")
	return result


spec = [
	('state'         , numba.int8[:,:]),
	('territories'   , numba.int8[:,:]),
	('active_ppl'    , numba.int8[:,:]),
	('declined_ppl'  , numba.int8[:,:]),
	('scores'    	 , numba.int8[:,:]),
	('people_deck'	 , numba.int8[:,:]),
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
		for i in range(len(descr)):
			if descr[i][3]:
				nb_primit = initial_nb_people[PRIMIT]
				self.territories[i,:] = [nb_primit, PRIMIT, 0, -1, 0]
			else:
				self.territories[i,:] = [0        , NOPPL , 0, -1, 0]

		# Init deck of people and draw for P0 and P1
		self._init_deck()
		chosen_ppl = self.people_deck[0, :]
		self.active_ppl[0,:]   = [chosen_ppl[0], chosen_ppl[1], NEW_TURN_STARTED , 0, chosen_ppl[4]]
		self._update_deck_after_chose(0)
		chosen_ppl = self.people_deck[0, :]
		self.active_ppl[1,:]   = [chosen_ppl[0], chosen_ppl[1], WAITING_OTHER_PL , 0, chosen_ppl[4]]
		self._update_deck_after_chose(0)

		self.declined_ppl[0,:] = [0, NOPPL , 0, 0, 0]
		self.declined_ppl[1,:] = [0, NOPPL , 0, 0, 0]

		# First round is round #1
		self._update_round()
	
	def copy_state(self, state, copy_or_not):
		if self.state is state and not copy_or_not:
			return
		self.state = state.copy() if copy_or_not else state

		self.territories  = self.state[0                        :NB_AREAS                 ,:]
		self.active_ppl   = self.state[NB_AREAS                 :NB_AREAS+  NUMBER_PLAYERS,:]
		self.declined_ppl = self.state[NB_AREAS+  NUMBER_PLAYERS:NB_AREAS+2*NUMBER_PLAYERS,:]
		self.scores       = self.state[NB_AREAS+2*NUMBER_PLAYERS:NB_AREAS+3*NUMBER_PLAYERS,:]
		self.people_deck  = self.state[NB_AREAS+3*NUMBER_PLAYERS:NB_AREAS+3*NUMBER_PLAYERS+DECK_SIZE+1,:]

	def valid_moves(self, player):
		result = np.zeros(10, dtype=np.bool_)
		return result

	def make_move(self, move, player, deterministic):
		return 1-player

	def get_state(self):
		return self.state

	def get_round(self):
		return self.scores[0,1]

	def check_end_game(self, next_player):
		if self.get_round() <= 10:
			return np.array([0, 0], dtype=np.float32) # No winner yet

		# Game is ended
		scores = self.scores[:, 0]
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
		current_ppl = self._current_ppl(player)

		# Check that player has a valid active ppl:
		if current_ppl[1] == NOPPL:
			return valids

		# Attack permitted when it's the time
		if current_ppl[2] not in [NEW_TURN_STARTED, JUST_ATTACKED, JUST_ABANDONED]:
			return valids

		territories_of_player = self._are_occupied_by(current_ppl)
		# Check that player could have at least 1 player in hand
		nb_ppl_of_player = current_ppl[0]
		if current_ppl[2] == JUST_ATTACKED:
			# Forbid to continue if 0 ppl left in hand
			if nb_ppl_of_player <= 0:
				return valids
		else:
			# 1st attck so simulate redeploy: add people from the board, except 1 per territory
			nb_ppl_to_get_from_board = np.dot(np.maximum(self.territories[:,0]-1,0), territories_of_player)
			nb_ppl_of_player += nb_ppl_to_get_from_board
			if current_ppl[1] == AMAZON:
				nb_ppl_of_player += 4

		for area in range(NB_AREAS):
			valids[area] = self._valid_attack_area(player, area, current_ppl, nb_ppl_of_player, territories_of_player)

		return valids

	def _valid_attack_area(self, player, area, current_ppl, nb_ppl_of_player, territories_of_player):
		# No attack on water
		if descr[area][0] == WATER:
			return False

		# No attack on current people
		if self._is_occupied_by(area, current_ppl):
			return False

		# Check no immunity
		if self.territories[area, 2] >= IMMUNE_CONQUEST:
			return False

		# Check that player has a chance to win	
		minimum_ppl_for_attack = self._minimum_ppl_for_attack(area, current_ppl)
		if nb_ppl_of_player + MAX_DICE < minimum_ppl_for_attack:
			return False

		# Check that territory is close to another owned territory or is on the edge
		nb_territories_of_player = np.count_nonzero(territories_of_player)
		if nb_territories_of_player == 0:
			area_is_on_edge = descr[area][2]
			if current_ppl[1] != HALFLING and not area_is_on_edge:
				return False
		else:
			neighbor_areas = connexity_matrix[area]
			if not np.any(np.logical_and(neighbor_areas, territories_of_player)):
				return False

		return True

	def _do_attack(self, player, area):
		current_ppl = self._current_ppl(player)

		# Prepare people if 1st action of the turn
		if current_ppl[2] != JUST_ATTACKED:
			if current_ppl[1] == AMAZON:
				print(f'Bonus Amazone {current_ppl[0]} --> {current_ppl[0]+4}')
				current_ppl[0] += 4
				current_ppl[4]  = 4
			self._gather_current_ppl_but_one(current_ppl)

		nb_ppl_of_player, type_ppl_of_player = current_ppl[0], current_ppl[1]
		minimum_ppl_for_attack = self._minimum_ppl_for_attack(area, current_ppl)

		# Use dice if people are needed
		use_dice = (nb_ppl_of_player < minimum_ppl_for_attack)
		if use_dice:
			dice = DICE_VALUES[3]
			if nb_ppl_of_player + dice < minimum_ppl_for_attack:
				print(f'  Using dice, random value is {dice} but fails')
				self._switch_from_attack_to_deploy(current_ppl)
				return
			print(f'  Using dice, random value is {dice} and succeed')
			nb_attacking_ppl = nb_ppl_of_player
		else:
			nb_attacking_ppl = minimum_ppl_for_attack

		# Attack is successful

		# Update loser and winner
		self._switch_territory_from_loser_to_winner(area, player, current_ppl, nb_attacking_ppl)

		# Update winner's status
		current_ppl[2] = JUST_ATTACKED
		# Redeploy if that was last attack
		if current_ppl[0] == 0 or use_dice:
			self._switch_from_attack_to_deploy(current_ppl)

	def _valids_redeploy(self, player):
		valids = np.zeros(NB_AREAS + MAX_REDEPLOY, dtype=np.bool_)
		current_ppl = self._current_ppl(player)

		# Check that it is time
		if current_ppl[2] not in [TO_START_REDEPLOY, TO_REDEPLOY]:
			if current_ppl[2] in [JUST_DECLINED, WAITING_OTHER_PL, NEED_ABANDON]:
				return valids

		# Check there is at least one active territory
		territories = self._are_occupied_by(current_ppl)
		nb_territories = np.count_nonzero(territories)
		if nb_territories == 0:
			# If no other option, then allow to skip redeploy
			valids[0] = True
			return valids

		# Check that player has still some ppl to deploy
		if current_ppl[2] == TO_REDEPLOY:
			how_many_ppl_available = current_ppl[0]
		else:
			how_many_ppl_available = self._ppl_virtually_available(current_ppl, territories)
			if current_ppl[1] == AMAZON:
				how_many_ppl_available -= current_ppl[4]
		if how_many_ppl_available <= 0:
			# If no other option, then allow to skip redeploy
			valids[0] = True
			return valids

		for ppl_to_deploy in range(1, MAX_REDEPLOY):
			valids[ppl_to_deploy] = self._valid_redeploy_on_each(player, ppl_to_deploy, how_many_ppl_available, nb_territories)
		for area in range(NB_AREAS):
			valids[MAX_REDEPLOY + area] = self._valid_redeploy_area(player, area, territories)

		if not valids.any():
			# If no other option, then allow to skip redeploy
			valids[0] = True

		return valids

	def _valid_redeploy_area(self, player, area, territories):
		# Check that this territory is owned
		if not territories[area]:
			return False
		return True

	def _valid_redeploy_on_each(self, player, how_many_to_deploy, how_many_ppl_available, nb_territories_of_player):
		if how_many_ppl_available < how_many_to_deploy*nb_territories_of_player:
			return False
		return True

	def _do_redeploy(self, player, param):
		current_ppl = self._current_ppl(player)

		if param == 0: # Special case, skip redeploy
			# Remove Amazon additional ppl before
			if current_ppl[1] == AMAZON and current_ppl[2] != TO_REDEPLOY:
				if current_ppl[0] < current_ppl[4]:
					current_ppl[2] = NEED_ABANDON
					return
				else:
					print(f'Removing {current_ppl[4]} Amazons {current_ppl[0]} --> {current_ppl[0]-current_ppl[4]}')
					current_ppl[0] -= current_ppl[4]
					current_ppl[4] = 0
			self._score_and_switch_to_next(player, current_ppl)
			return

		if current_ppl[2] != TO_REDEPLOY:
			self._gather_current_ppl_but_one(current_ppl, redeploy=True)
			if current_ppl[1] == AMAZON:
				print(f'Removing {current_ppl[4]} Amazons {current_ppl[0]} --> {current_ppl[0]-current_ppl[4]}')
				current_ppl[0] -= current_ppl[4]
				current_ppl[4] = 0
				assert(current_ppl[0] >= 0)
			current_ppl[2] = TO_REDEPLOY

		if param < MAX_REDEPLOY:
			# Deploy X ppl on all active areas
			how_many_to_deploy = param

			territories_of_player = self._are_occupied_by(current_ppl)
			nb_territories_of_player = np.count_nonzero(territories_of_player)
			current_ppl[0] -= how_many_to_deploy*nb_territories_of_player
			assert(current_ppl[0] >= 0)
			self.territories[:, 0] += how_many_to_deploy*territories_of_player
		else:
			# Deploy 1 ppl on 1 area
			area = param - MAX_REDEPLOY

			current_ppl[0]            -= 1
			self.territories[area, 0] += 1

		# Trigger end of turn if no more to redeploy
		if current_ppl[0] == 0:
			self._score_and_switch_to_next(player, current_ppl)

	def _valid_decline(self, player):
		# Going to decline permitted only on 1st move
		if self.active_ppl[player, 2] != NEW_TURN_STARTED:
			return False
		if self.active_ppl[player, 1] == NOPPL:
			return False
		return True

	def _do_decline(self, player):
		current_ppl = self.active_ppl[player, :]
		# Remove previous declined ppl from the board
		for area in range(NB_AREAS):
			if self._is_occupied_by(area, self.declined_ppl[player, :]):
				self.territories[area] = [0, NOPPL, 0, -1, 0]

		# Move ppl to decline and keep only 1 ppl per territory
		if current_ppl[1] != GHOUL:
			self._gather_current_ppl_but_one(current_ppl)
			self.declined_ppl[player, :] = [0             , current_ppl[1], 0               , 0, 0]
		else:
			self.declined_ppl[player, :] = [current_ppl[0], current_ppl[1], WAITING_OTHER_PL, 0, 0]
		
		# Flip back ppl tokens on the board and remove defense
		for area in range(NB_AREAS):
			if self.territories[area, 1] == self.declined_ppl[player, 1]:
				self.territories[area, 1] = -self.declined_ppl[player, 1]
				self.territories[area, 2] = 0
		self.declined_ppl[player, 1] = -self.declined_ppl[player, 1]

		current_ppl[:] = [0, NOPPL, WAITING_OTHER_PL, 0, 0]
		self._score_and_switch_to_next(player, current_ppl, force_other_player=True)

	def _valids_choose_ppl(self, player):
		valids = np.zeros(DECK_SIZE, dtype=np.bool_)

		# Check that it is time
		if self.active_ppl[player, 2] != NEW_TURN_STARTED:
			return valids
		# Check that player hasn't a player yet
		if self.active_ppl[player, 1] != NOPPL:
			return valids

		for index in range(DECK_SIZE):
			# Check that index is valid
		 	valids[index] = (self.people_deck[index, 1] != NOPPL)

		return valids

	def _do_choose_ppl(self, player, index):
		self.active_ppl[player,:] = self.people_deck[index, :]
		self.active_ppl[player,2] = JUST_ATTACKED
		self._update_deck_after_chose(index)

	def _valids_abandon(self, player):
		valids = np.zeros(NB_AREAS, dtype=np.bool_)
		current_ppl = self._current_ppl(player)

		# To avoid infinite loops, allow to abandon area only at the beginning of the turn
		if current_ppl[2] not in [NEW_TURN_STARTED, NEED_ABANDON]:
			return valids
		# Cant abandon if player doesn't have any active people
		if current_ppl[1] == NOPPL:
			return valids

		for area in range(NB_AREAS):
			valids[area] = self._is_occupied_by(area, current_ppl)

		return valids

	def _do_abandon(self, player, area):
		current_ppl = self._current_ppl(player)
		self._leave_area(area)

		if current_ppl[2] == NEED_ABANDON: # Case of amazon not able to give back 4 ppl
			how_many_ppl_available = self._ppl_virtually_available(current_ppl)
			current_ppl[2] = NEED_ABANDON if how_many_ppl_available < current_ppl[4] else TO_START_REDEPLOY
		else:
			current_ppl[2] = JUST_ABANDONED

	###########################################################################

	def _current_ppl(self, player):
		if 0 < self.declined_ppl[player, 2] < WAITING_OTHER_PL:
			return self.declined_ppl[player, :]
		if 0 < self.active_ppl[player, 2] < WAITING_OTHER_PL:
			return self.active_ppl[player, :]
		raise Exception(f'No ppl to play for P{player}')

	def _ppl_owner_of(self, area):
		owner = self.territories[area,3]
		if owner < 0:
			return None
		if self.active_ppl[owner, 1] == self.territories[area, 1]:
			return self.active_ppl[owner, :]
		if self.declined_ppl[owner, 1] == self.territories[area, 1]:
			return self.declined_ppl[owner, :]
		raise Exception(f'Could not find which ppl this area belongs ({area=} {owner=} {self.territories[area, 1]=})')

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
		minimum_ppl_for_attack = self.territories[area,0] + self.territories[area, 2] + 2

		# Malus if: mountain, troll (even in decline)
		if descr[area][0] == MOUNTAIN:
			minimum_ppl_for_attack += 1
		if abs(self.territories[area, 1]) == TROLL:
			minimum_ppl_for_attack += 1

		# Bonus if: triton + at_edge, giant + border of mountain
		if current_ppl[1] == TRITON and descr[area][2]:
			minimum_ppl_for_attack = max(minimum_ppl_for_attack - 1, 1)
		if current_ppl[1] == GIANT  and self._is_area_border_of(area, MOUNTAIN):
			minimum_ppl_for_attack = max(minimum_ppl_for_attack - 1, 1)

		return minimum_ppl_for_attack

	def _leave_area(self, area):
		# Give back ppl to owner
		self._ppl_owner_of(area)[0] += self.territories[area, 0]

		# Make the area empty
		self.territories[area,:] = [0, NOPPL, 0, -1, 0]

	def _switch_territory_from_loser_to_winner(self, area, player, winner_ppl, nb_attacking_ppl):
		nb_initial_ppl = self.territories[area, 0]

		# Give back people to the loser (if any)
		loser_ppl = self._ppl_owner_of(area)
		if loser_ppl is not None:
			assert(nb_initial_ppl > 0)
			# Lose 1 ppl unless you are elf and active
			nb_ppl_to_lose = 1 if self.territories[area,1] != ELF else 0
			loser_ppl[0] += self.territories[area,0] - nb_ppl_to_lose

			loser_territories = self._are_occupied_by(loser_ppl)
			total_ppl = np.dot(self.territories[:,0], loser_territories) + loser_ppl[0]
			if total_ppl <= 0:
				print('Last people was lost')
				loser_ppl[:] = [0, NOPPL, WAITING_OTHER_PL, 0, 0]

		# Install people from the winner
		self.territories[area,0] = nb_attacking_ppl
		self.territories[area,1] = winner_ppl[1]
		self.territories[area,2] = 0
		self.territories[area,3] = player
		self.territories[area,4] = 0
		winner_ppl[0] -= nb_attacking_ppl
		assert(winner_ppl[0] >= 0)
		# Add specific tokens
		if winner_ppl[1] == HALFLING and winner_ppl[4] > 0:
			self.territories[area, 2] = FULL_IMMUNITY
			winner_ppl[4] -= 1

		# Update #NETWDT
		if nb_initial_ppl > 0:
			winner_ppl[3] += 1

	def _gather_current_ppl_but_one(self, current_ppl, redeploy=False):
		# Gather all active people in player's hand, leaving only 1 on each territory
		# print(f'Prepare / redeploy P{player}:', end='')
		for area in range(NB_AREAS):
			if self._is_occupied_by(area, current_ppl):
				nb_ppl_to_gather = max(self.territories[area,0] - 1, 0)
				if nb_ppl_to_gather > 0:
					self.territories[area,0] -= nb_ppl_to_gather
					current_ppl[0]           += nb_ppl_to_gather
					# print(f' {nb_ppl_to_gather}ppl on area{area}', end='')

		# If redeploy, additional people for skeleton (up to 20)
		if redeploy and current_ppl[1] == SKELETON:
			current_ppl[0] += (current_ppl[3] // 2)
			current_ppl[0] = min(MAX_SKELETONS, current_ppl[0])
		# print('')

	def _ppl_virtually_available(self, current_ppl, player_territories=None):
		if player_territories is None:
			player_territories = self._are_occupied_by(current_ppl)

		how_many_ppl_available = current_ppl[0]
		# Simulate redeploy: add people on the boards, except 1 per territory
		how_many_ppl_available += np.dot(np.maximum(self.territories[:,0]-1,0), player_territories)

		# Additional people for skeleton (up to 20)
		if current_ppl[1] == SKELETON:
			how_many_ppl_available += (current_ppl[3] // 2)
			how_many_ppl_available = min(MAX_SKELETONS, how_many_ppl_available)

		return how_many_ppl_available

	def _switch_from_attack_to_deploy(self, current_ppl):
		current_ppl[2] = TO_START_REDEPLOY
		if current_ppl[1] == AMAZON:
			how_many_ppl_available = self._ppl_virtually_available(current_ppl)
			if how_many_ppl_available < current_ppl[4]:
				print(f'Amazons need to abandon some areas ({how_many_ppl_available})')
				current_ppl[2] = NEED_ABANDON
			# else:
				# print(f'No need to abandon areas ({how_many_ppl_available})')

	def _score_and_switch_to_next(self, player, current_ppl, new_status_of_cur_player=WAITING_OTHER_PL, force_other_player=False):
		current_ppl[2] = new_status_of_cur_player

		if current_ppl[1] > 0 or force_other_player:		# Next turn is for next player
			self._update_score(player)

			next_player = (player+1) % NUMBER_PLAYERS
			if next_player == 0:
				self._update_round()
			if self.declined_ppl[next_player, 1] == -GHOUL:
				next_ppl = self.declined_ppl[next_player, :]
			else:
				next_ppl = self.active_ppl[next_player, :]
		else:  												# Next turn is for same player
			print('Same player to play with its active ppl now')
			next_player = player
			next_ppl = self.active_ppl[player, :]

		if next_ppl[2] == WAITING_OTHER_PL:
			next_ppl[2] = NEW_TURN_STARTED
		else:
			print(f'Wrong status for {next_player}: {next_ppl[2]}')
			raise Exception

	def _update_score(self, player):
		owned_areas = (self.territories[:,3] == player)
		score_for_this_turn = 0

		# Iterate on areas and count score
		for area in [area for area in range(NB_AREAS) if owned_areas[area]]:
			score_for_this_turn += self.territories[area, 0]
			# +1 point if: dwarf + mine (even in decline), human + field, wizard + magic
			if descr[area][1] == MINE     and abs(self.territories[area, 1]) == DWARF:
				score_for_this_turn += 1
			if descr[area][0] == FARMLAND and     self.territories[area, 1]  == HUMAN:
				score_for_this_turn += 1
			if descr[area][1] == MAGIC    and     self.territories[area, 1]  == WIZARD:
				score_for_this_turn += 1

		# +1 point if: orc + NETWDT
		if self.active_ppl[player, 1] == ORC:
			score_for_this_turn += self.active_ppl[player, 3]

		self.scores[player][0] += score_for_this_turn

		# Reset NETWDT
		self.active_ppl[player, 3] = 0
		self.declined_ppl[player, 3] = 0

	def _update_round(self):
		self.scores[:,1] += 1

	def _init_deck(self):
		# All people available except NOPPL and PRIMIT
		available_people = np.ones(PRIMIT, dtype=np.int8)
		available_people[NOPPL] = False

		# Draw 6 ppl randomly
		for i in range(DECK_SIZE):
			# chosen_ppl = my_random_choice(available_people / available_people.sum())
			chosen_ppl = [GHOUL, AMAZON, HALFLING, TROLL, GIANT, TRITON, HUMAN, WIZARD, DWARF, ELF][i]
			self.people_deck[i, :] = [initial_nb_people[chosen_ppl], chosen_ppl, 0, 0, initial_tokens[chosen_ppl]]
			available_people[chosen_ppl] = False

		# Update bitfield
		self.people_deck[DECK_SIZE, :2] = divmod(my_packbits(available_people), 256)

	def _update_deck_after_chose(self, index):
		# Read bitfield
		available_people = my_unpackbits(256*self.people_deck[DECK_SIZE, 0] + self.people_deck[DECK_SIZE, 1])

		# Delete people #item and shift others upwards
		self.people_deck[index:DECK_SIZE-1, :] = self.people_deck[index+1:DECK_SIZE, :]
		# Draw a new people for last combination
		chosen_ppl = my_random_choice(available_people / available_people.sum())
		self.people_deck[DECK_SIZE-1, :] = [initial_nb_people[chosen_ppl], chosen_ppl, 0, 0, initial_tokens[chosen_ppl]]
		available_people[chosen_ppl] = False

		# Update back the bitfield
		self.people_deck[DECK_SIZE, :2] = divmod(my_packbits(available_people), 256)

###############################################################################
import random

b = Board(NUMBER_PLAYERS)
print_board(b)
print()

def play_one_turn():
	who_can_play = np.logical_or(
		np.logical_and(b.active_ppl  [:, 2] > 0, b.active_ppl  [:, 2] < WAITING_OTHER_PL),
		np.logical_and(b.declined_ppl[:, 2] > 0, b.declined_ppl[:, 2] < WAITING_OTHER_PL)
	)
	p = who_can_play.nonzero()[0].item()
	print(f'Player is now P{p}')

	while b.active_ppl[p, 2] < WAITING_OTHER_PL or b.declined_ppl[p, 2] not in [0, WAITING_OTHER_PL]:
		valids_attack    = b._valids_attack(player=p)
		valids_abandon   = b._valids_abandon(player=p)
		valids_redeploy  = b._valids_redeploy(player=p)
		valids_choose    = b._valids_choose_ppl(player=p)
		valid_decline    = b._valid_decline(player=p)
		print_valids(p, valids_attack, valids_abandon, valids_redeploy, valids_choose, valid_decline)

		if any(valids_attack) or any(valids_abandon) or valid_decline:
			values = np.concatenate((valids_attack.nonzero()[0], valids_abandon.nonzero()[0] + NB_AREAS, ([2*NB_AREAS] if valid_decline else [])), axis=None)
			dice = np.random.choice(values.astype(np.int64))

			if valid_decline and not any(valids_attack) and not any(valids_abandon):
				breakpoint()

			if dice < NB_AREAS:
				area = dice
				print(f'Attacking area {area}')
				b._do_attack(player=p, area=area)
			elif dice < 2*NB_AREAS:
				area = dice - NB_AREAS
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