import sys
sys.path.append('../../')
from GenericNNetWrapper import GenericNNetWrapper
from .AkropolisNNet import AkropolisNNet as nn_model
from .AkropolisConstants import CITY_SIZE, N_PLAYERS

class NNetWrapper(GenericNNetWrapper):
	def init_nnet(self, game, nn_args):
		self.nnet = nn_model(game, nn_args)

	def reshape_boards(self, numpy_boards):
		# Some game needs to reshape boards before being an input of NNet
		return numpy_boards.reshape(-1, CITY_SIZE*CITY_SIZE, 2*N_PLAYERS+2)