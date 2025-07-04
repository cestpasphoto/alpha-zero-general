import sys
sys.path.append('../../')
from GenericNNetWrapper import GenericNNetWrapper
from .AkropolisNNet import AkropolisNNet as nn_model

class NNetWrapper(GenericNNetWrapper):
	def init_nnet(self, game, nn_args):
		self.nnet = nn_model(game, nn_args)