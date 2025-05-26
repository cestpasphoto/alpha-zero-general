import sys
sys.path.append('../../')
from GenericNNetWrapper import GenericNNetWrapper
from .BotanikNNet import BotanikNNet as nn_model
from .BotanikConstants import NB_ROWS_FOR_MACH

class NNetWrapper(GenericNNetWrapper):
	def init_nnet(self, game, nn_args):
		self.nnet = nn_model(game, nn_args)
