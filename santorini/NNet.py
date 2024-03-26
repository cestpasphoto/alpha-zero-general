import sys

sys.path.append('../../')
from GenericNNetWrapper import GenericNNetWrapper
from SantoriniNNet import SantoriniNNet as nn_model


class NNetWrapper(GenericNNetWrapper):
	def init_nnet(self, game, nn_args):
		self.nnet = nn_model(game, nn_args)

	def reshape_boards(self, numpy_boards):
		# Some game needs to reshape boards before being an input of NNet
		return numpy_boards.reshape(-1, 25, 3)
