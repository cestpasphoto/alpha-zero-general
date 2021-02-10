import torch
import torch.nn as nn
import torch.nn.functional as F

class SplendorNNet(nn.Module):
	def __init__(self, game, args):
		# game params
		self.nb_vect, self.vect_dim = game.getBoardSize()
		self.action_size = game.getActionSize()
		self.args = args

		super(SplendorNNet, self).__init__()
		dense2d, dense1d = args['dense2d'], args['dense1d']

		def _init(m):
			if type(m) == nn.Linear:
				nn.init.kaiming_uniform_(m.weight)
				m.bias.data.fill_(0.01)
		
		self.linear2D = nn.ModuleList([nn.Linear(prev, cur) for prev, cur in zip([self.nb_vect] +dense2d, dense2d)])
		self.linear2D.apply(_init)
		self.linear1D = nn.ModuleList([nn.Linear(prev, cur) for prev, cur in zip([dense2d[-1]*9]+dense1d, dense1d)])
		self.linear1D.apply(_init)
		self.batch_norms = nn.ModuleList([nn.BatchNorm1d(cur) for cur in dense1d])

		self.output_layer_PI = nn.Linear(dense1d[-1], self.action_size)
		self.output_layer_V  = nn.Linear(dense1d[-1], 1)
		self.maxpool = nn.MaxPool2d((5,1))
		self.avgpool = nn.AvgPool2d((5,1))

	def forward(self, input_data, valid_actions):
		if len(input_data.shape) == 3:
			x = input_data.permute(0,2,1).view(-1, self.vect_dim, self.nb_vect)
		else:
			x = input_data.permute(1,0).view(-1, self.vect_dim, self.nb_vect)
		
		for layer2d in self.linear2D:
			x = F.relu(layer2d(x))
		
		x_5lay,_ = x.split([5,2], 1)
		x_max    = self.maxpool(x_5lay)
		x_avg    = self.avgpool(x_5lay)
		x = nn.Flatten()(torch.cat((x_max, x_avg, x), 1))

		for layer1d, bn in zip(self.linear1D, self.batch_norms):
			x = F.dropout(F.relu(bn(layer1d(x))), p=self.args['dropout'], training=self.training)
		
		v = self.output_layer_V(x)
		pi = torch.where(valid_actions, self.output_layer_PI(x), torch.FloatTensor([-1e8]))
		
		return F.log_softmax(pi, dim=1), torch.tanh(v)
