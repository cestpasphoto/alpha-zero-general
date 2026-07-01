import sys
sys.path.append('../../')
from GenericNNetWrapper import GenericNNetWrapper
from .SmallworldNNet import SmallworldNNet as nn_model
from .SmallworldNNet_graph import SmallworldGraphNNet

class NNetWrapper(GenericNNetWrapper):
    def init_nnet(self, game, nn_args):
        if nn_args['nn_version'] in (72, 73, 74):
            self.nnet = SmallworldGraphNNet(game, nn_args)
        else:
            self.nnet = nn_model(game, nn_args)