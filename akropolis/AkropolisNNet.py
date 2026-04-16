import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import _make_divisible
from torchvision.models.mobilenetv3 import InvertedResidualConfig, InvertedResidual

from .AkropolisConstants import N_COLORS, CITY_SIZE, CITY_AREA, CONSTR_SITE_SIZE, CODES_LIST

# --- HELPER CLASSES FOR NEW ARCHITECTURES ---

class FiLMLayer(nn.Module):
	"""Feature-wise Linear Modulation"""
	def __init__(self, channels):
		super().__init__()
		self.channels = channels

	def forward(self, x, gamma, beta):
		# x: (N, C, H, W), gamma/beta: (N, C)
		gamma = gamma.unsqueeze(-1).unsqueeze(-1)
		beta = beta.unsqueeze(-1).unsqueeze(-1)
		return x * gamma + beta

class GlobalContextMLP(nn.Module):
	"""Processes dynamic heterogeneous global data into a fixed-size context vector"""
	def __init__(self, num_players, embed_dim, out_dim):
		super().__init__()
		self.embed = nn.Embedding(num_embeddings=len(CODES_LIST), embedding_dim=embed_dim)
		
		# Calculate input size dynamically based on num_players
		# Scores/Districts/Plazas: 3 * num_players * N_COLORS
		# Misc: 2 (round, remaining stacks)
		# Constr Site: CONSTR_SITE_SIZE * 3 hexes * embed_dim
		self.flat_size = (3 * num_players * N_COLORS) + 2 + (CONSTR_SITE_SIZE * 3 * embed_dim)
		
		self.mlp = nn.Sequential(
			nn.Linear(self.flat_size, 64),
			nn.BatchNorm1d(64),
			nn.Hardswish(),
			nn.Linear(64, out_dim)
		)

	def forward(self, scores_data, misc_data, constrs_site):
		# Embed construction site descriptions
		constrs_long = constrs_site.clamp(min=0., max=len(CODES_LIST)-1).long()
		c_emb = self.embed(constrs_long) # (N, CS, 3, D)
		
		# Flatten everything
		flat_scores = scores_data.flatten(1) # (N, 3 * num_players * N_COLORS)
		flat_c_emb = c_emb.flatten(1)        # (N, CS * 3 * D)
		
		x = torch.cat([flat_scores, misc_data, flat_c_emb], dim=1)
		return self.mlp(x), c_emb

class AkropolisNNet(nn.Module):
	def __init__(self, game, args):	
		self.board_size = game.getBoardSize()
		self.action_size = game.getActionSize()
		self.num_players = game.num_players
		self.args = args
		self.version = args['nn_version']
		super(AkropolisNNet, self).__init__()

		def inverted_residual(input_ch, expanded_ch, out_ch, use_se, activation, kernel=3, stride=1, dilation=1, width_mult=1):
			return InvertedResidual(InvertedResidualConfig(input_ch, kernel, expanded_ch, out_ch, use_se, activation, stride, dilation, width_mult), nn.BatchNorm2d)

		def filters_for_boards(input_ch, expanded_ch, out_ch, depth):
			filters  = [nn.Conv2d(input_ch, input_ch, kernel_size=3, padding=1, bias=False)]
			filters += [inverted_residual(input_ch, expanded_ch, input_ch, False, "RE") for i in range(depth//2)]
			filters += [inverted_residual(input_ch, expanded_ch, out_ch, False, "RE")]
			filters += [inverted_residual(out_ch, expanded_ch, out_ch, True, "HS") for i in range(depth//2)]
			filters += [nn.Flatten()]
			return nn.Sequential(*filters)

		if self.version == 1: # Ultra simple NN
			num_filters = 100
			input_size = 3*self.num_players*N_COLORS + 3*CONSTR_SITE_SIZE + 2 + 2*self.num_players*CITY_AREA
			self.trunk_1d = nn.Sequential(
				nn.Linear(input_size, num_filters),
				nn.BatchNorm1d(num_filters),
				nn.Hardswish(),
			)
			self.final_layers_V = nn.Sequential(
				nn.Linear(num_filters, num_filters),
				nn.BatchNorm1d(num_filters),
				nn.ReLU(),
				nn.Linear(num_filters, 1),
			)
			self.final_layers_PI = nn.Sequential(
				nn.Linear(num_filters, self.action_size),
			)
		elif self.version in [30, 31, 32]: # Less simple version using Embedding
			constants = {
				#    D   T   S   G  B  r
				30: (3, 32, 16, 8 , 8 , 16),
				31: (3, 32, 16, 8 , 8 , 16),
				32: (3, 32, 16, 8 , 16, 16),
			}
			D, T, S, G, B, r = constants[self.version]
			self.embed = nn.Embedding(num_embeddings=len(CODES_LIST), embedding_dim=D)
			self.conv1d_constr = nn.Sequential(
				nn.Conv1d(D, T, kernel_size=3, stride=1, padding=0),
				# batchnorm not relevant since we put some channel data into the batch dimension
				nn.Hardswish(),
			)
			self.dense_scores = nn.Sequential(
				nn.Linear(N_COLORS*3*self.num_players, S),
			)
			self.conv2d_boards = nn.Sequential(
				nn.Conv2d(D+2, B, kernel_size=3, padding=1),
				nn.BatchNorm2d(B),
				nn.Hardswish(),
				nn.Conv2d(B, B, kernel_size=3, padding=1),
				nn.BatchNorm2d(B),
				nn.Hardswish(),
			)
			self.dense_globs = nn.Sequential(
				nn.Linear(2, G),
				nn.BatchNorm1d(G),
				nn.Hardswish(),
			)
	
			# V head
			self.final_layers_V = nn.Sequential(
				nn.Flatten(1),
				nn.Linear(CONSTR_SITE_SIZE*(T+S+G), r),
				nn.BatchNorm1d(r),
				nn.Hardswish(),
				nn.Linear(r, r),
				nn.Hardswish(),
				nn.Linear(r, self.num_players),
			)

			# PI head
			self.proj_i = nn.Linear(T+S+G, r)
			self.proj_p = nn.Sequential(
				inverted_residual(self.num_players*B+G+S, 2*r, r, True, "HS", kernel=1),
			)
			if self.version == 30:
				self.U_o = nn.Parameter(torch.randn(6, r))
				nn.init.xavier_uniform_(self.U_o)			
			else:
				self.proj_o = nn.Linear(T+G+S, 6 * r)
			self.b1n = nn.Sequential(
				nn.BatchNorm1d(r),
				nn.Hardswish(),
			)

		# =====================================================================
		# V40: FiLM-Conditioned MobileNet (Attention Contextuelle)
		# =====================================================================
		elif self.version == 40:
			D = 8         # Embedding dimension for hex descriptions
			C_sp = 24     # Spatial channels (kept very small for < 2 MFLOPs)
			C_ctx = 64    # Global context dimension
			
			self.embed = nn.Embedding(num_embeddings=len(CODES_LIST), embedding_dim=D)
			self.global_extractor = GlobalContextMLP(self.num_players, D, C_ctx)
			
			# Spatial stem for Player 0 only (Embedding D + Height 1)
			self.stem = nn.Sequential(
				nn.Conv2d(D + 1, C_sp, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(C_sp),
				nn.Hardswish()
			)
			
			# Spatial Blocks with FiLM
			self.block1 = inverted_residual(C_sp, C_sp*3, C_sp, False, "RE")
			self.film1_gamma = nn.Linear(C_ctx, C_sp)
			self.film1_beta  = nn.Linear(C_ctx, C_sp)
			self.film1 = FiLMLayer(C_sp)
			
			self.block2 = inverted_residual(C_sp, C_sp*3, C_sp, True, "HS")
			self.film2_gamma = nn.Linear(C_ctx, C_sp)
			self.film2_beta  = nn.Linear(C_ctx, C_sp)
			self.film2 = FiLMLayer(C_sp)

			# Policy Head (Einsum approach, factorized)
			self.proj_board = nn.Conv2d(C_sp, C_sp, kernel_size=1)
			self.proj_tile = nn.Linear(3 * D, C_sp)
			self.proj_orient = nn.Linear(C_sp, 6 * C_sp)
			
			# Value Head (Dynamic output size = num_players)
			self.val_head = nn.Sequential(
				nn.Linear(C_sp + C_ctx, 32),
				nn.Hardswish(),
				nn.Linear(32, self.num_players)
			)

		# =====================================================================
		# V41: Early-Broadcast Bottleneck (Compression Brutale)
		# =====================================================================
		elif self.version == 41:
			D = 8
			C_ctx = 16    # Tiny broadcast vector
			C_sp = 24     # Spatial channels

			self.embed = nn.Embedding(num_embeddings=len(CODES_LIST), embedding_dim=D)
			self.global_extractor = GlobalContextMLP(self.num_players, D, C_ctx)
			
			# Bottleneck mixing Broadcasted Context (C_ctx) + Spatial P0 (D + 1)
			self.bottleneck = nn.Sequential(
				nn.Conv2d(D + 1 + C_ctx, C_sp, kernel_size=1, bias=False),
				nn.BatchNorm2d(C_sp),
				nn.Hardswish()
			)
			
			# Standard Spatial Blocks
			self.spatial_trunk = nn.Sequential(
				inverted_residual(C_sp, C_sp*3, C_sp, False, "RE"),
				inverted_residual(C_sp, C_sp*3, C_sp, True, "HS"),
				inverted_residual(C_sp, C_sp*3, C_sp, True, "HS")
			)

			# Policy Head
			self.proj_board = nn.Conv2d(C_sp, C_sp, kernel_size=1)
			self.proj_tile = nn.Linear(3 * D, C_sp)
			self.proj_orient = nn.Linear(C_sp, 6 * C_sp)
			
			# Value Head
			self.val_head = nn.Sequential(
				nn.Linear(C_sp + C_ctx, 32),
				nn.Hardswish(),
				nn.Linear(32, self.num_players)
			)

		# =====================================================================
		# V42: Asymmetric Dual-Stream DEEP
		# =====================================================================
		elif self.version == 42:
			D = 16        # Increased to 16 for very rich categorical representation
			C_sp = 16     # Must be a multiple of 8. Kept at 16 to allow more depth.
			C_ctx = 64    # Global context dimension

			self.embed = nn.Embedding(num_embeddings=len(CODES_LIST), embedding_dim=D)
			self.global_extractor = GlobalContextMLP(self.num_players, D, C_ctx)
			
			# Deep analytical MLP for Global Data
			self.deep_ctx = nn.Sequential(
				nn.Linear(C_ctx, C_ctx),
				nn.Hardswish(),
				nn.Linear(C_ctx, C_ctx)
			)

			# Spatial Stream for Player 0
			# Deeper network (4 blocks) to maximize the receptive field on the 13x13 grid
			self.spatial_trunk = nn.Sequential(
				nn.Conv2d(D + 1, C_sp, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(C_sp),
				nn.Hardswish(),
				inverted_residual(C_sp, C_sp*3, C_sp, False, "RE"),
				inverted_residual(C_sp, C_sp*3, C_sp, True, "HS"),
				inverted_residual(C_sp, C_sp*3, C_sp, True, "HS"),
				inverted_residual(C_sp, C_sp*3, C_sp, True, "HS"),
			)

			# Policy Head
			self.proj_board = nn.Conv2d(C_sp, C_sp, kernel_size=1)
			self.proj_tile = nn.Linear(3 * D + C_ctx, C_sp) 
			self.proj_orient = nn.Linear(C_sp, 6 * C_sp)
			
			# Value Head
			self.val_head = nn.Sequential(
				nn.Linear(C_sp + C_ctx, 32),
				nn.Hardswish(),
				nn.Linear(32, self.num_players)
			)

		# =====================================================================
		# V50: Siamese Spatial Pooling (Independent opponent summary)
		# =====================================================================
		elif self.version == 50:
			D = 12        # Categorical embedding dimension
			C_sp = 16     # Spatial channels (kept at 16 to afford Siamese processing)
			C_ctx = 64    # Global context dimension

			self.embed = nn.Embedding(num_embeddings=len(CODES_LIST), embedding_dim=D)
			self.global_extractor = GlobalContextMLP(self.num_players, D, C_ctx)
			
			# SHARED spatial stem (used for Player 0 AND Opponents)
			self.shared_stem = nn.Sequential(
				nn.Conv2d(D + 1, C_sp, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(C_sp),
				nn.Hardswish(),
				inverted_residual(C_sp, C_sp*3, C_sp, False, "RE")
			)
			
			# Player 0 specific deeper trunk (for geometric planning)
			self.p0_trunk = nn.Sequential(
				inverted_residual(C_sp, C_sp*3, C_sp, True, "HS"),
				inverted_residual(C_sp, C_sp*3, C_sp, True, "HS")
			)

			# Deep context includes Global + Opponent Summary (Avg & Max pooled -> 2 * C_sp)
			self.deep_ctx = nn.Sequential(
				nn.Linear(C_ctx + (2 * C_sp), C_ctx),
				nn.Hardswish(),
				nn.Linear(C_ctx, C_ctx)
			)

			# Policy Head
			self.proj_board = nn.Conv2d(C_sp, C_sp, kernel_size=1)
			self.proj_tile = nn.Linear(3 * D + C_ctx, C_sp) 
			self.proj_orient = nn.Linear(C_sp, 6 * C_sp)
			
			# Value Head
			self.val_head = nn.Sequential(
				nn.Linear(C_sp + C_ctx, 32),
				nn.Hardswish(),
				nn.Linear(32, self.num_players)
			)

		# =====================================================================
		# V51: Opponent Threat Attention (Targeted evaluation)
		# =====================================================================
		elif self.version == 51:
			D = 12
			C_sp = 16     # Main spatial channels
			C_opp = 16    # Smaller channels for opponent spatial attention
			C_ctx = 64    

			self.embed = nn.Embedding(num_embeddings=len(CODES_LIST), embedding_dim=D)
			self.global_extractor = GlobalContextMLP(self.num_players, D, C_ctx)
			
			# Spatial Stream for Player 0
			self.spatial_trunk = nn.Sequential(
				nn.Conv2d(D + 1, C_sp, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(C_sp),
				nn.Hardswish(),
				inverted_residual(C_sp, C_sp*3, C_sp, False, "RE"),
				inverted_residual(C_sp, C_sp*3, C_sp, True, "HS"),
				inverted_residual(C_sp, C_sp*3, C_sp, True, "HS")
			)

			# Ultra-light Opponent Stem
			self.opp_stem = nn.Sequential(
				nn.Conv2d(D + 1, C_opp, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(C_opp),
				nn.Hardswish()
			)

			# Cross-Attention projections (Q=Tiles, K/V=Opponent Board)
			self.q_proj = nn.Linear(3 * D, C_opp)
			self.k_proj = nn.Conv2d(C_opp, C_opp, kernel_size=1)
			self.v_proj = nn.Conv2d(C_opp, C_opp, kernel_size=1)

			# Deep context
			self.deep_ctx = nn.Sequential(
				nn.Linear(C_ctx, C_ctx),
				nn.Hardswish()
			)

			# Policy Head (Fuses Tile + Context + Opponent Threat Vector)
			self.proj_board = nn.Conv2d(C_sp, C_sp, kernel_size=1)
			self.proj_tile = nn.Linear(3 * D + C_ctx + C_opp, C_sp) 
			self.proj_orient = nn.Linear(C_sp, 6 * C_sp)
			
			# Value Head
			self.val_head = nn.Sequential(
				nn.Linear(C_sp + C_ctx, 32),
				nn.Hardswish(),
				nn.Linear(32, self.num_players)
			)


		self.register_buffer('lowvalue', torch.FloatTensor([-1e8]))
		def _init(m):
			if type(m) == nn.Linear:
				nn.init.kaiming_uniform_(m.weight)
				nn.init.zeros_(m.bias)
			elif type(m) == nn.Sequential:
				for module in m:
					_init(module)
		for _, layer in self.__dict__.items():
			if isinstance(layer, nn.Module):
				layer.apply(_init)

	def forward(self, input_data, valid_actions):
		x = input_data.permute(0, 3, 1, 2)
		
		# Slicer
		split_input_data = x.split([self.num_players, self.num_players, self.num_players, 1 ,1], dim=1)
		boards_descr, boards_height, boards_tileID, per_pl_data, global_data = split_input_data
		
		# Extract Global Context variables
		scores_data = per_pl_data.squeeze(1)[:, :3*self.num_players, :N_COLORS]
		constrs_site = global_data.squeeze(1)[:, :CONSTR_SITE_SIZE, :3]
		misc_data = global_data.squeeze(1)[:, CONSTR_SITE_SIZE+1, :2]
		globals_data = global_data.squeeze(1)[:, CONSTR_SITE_SIZE+1, :2]

		if self.version in [50, 51]:
			# --- COMMON SPATIAL PREPARATION ---
			# Player 0
			descr_p0 = boards_descr[:, 0, :, :].clamp(min=0., max=len(CODES_LIST)-1).long()
			spatial_emb_p0 = self.embed(descr_p0).permute(0, 3, 1, 2)
			height_p0 = boards_height[:, 0, :, :].unsqueeze(1)
			spatial_p0 = torch.cat([spatial_emb_p0, height_p0], dim=1) # (N, D+1, 13, 13)

			# Opponents (Dynamic handling of multiple opponents)
			num_opp = self.num_players - 1
			descr_opp = boards_descr[:, 1:, :, :].clamp(min=0., max=len(CODES_LIST)-1).long() # (N, num_opp, 13, 13)
			spatial_emb_opp = self.embed(descr_opp).permute(0, 1, 4, 2, 3) # (N, num_opp, D, 13, 13)
			height_opp = boards_height[:, 1:, :, :].unsqueeze(2) # (N, num_opp, 1, 13, 13)
			spatial_opp = torch.cat([spatial_emb_opp, height_opp], dim=2) # (N, num_opp, D+1, 13, 13)
			
			# Flatten batch and opp dimensions for standard Conv2d processing
			N, _, C_in, H, W = spatial_opp.shape
			spatial_opp_flat = spatial_opp.view(N * num_opp, C_in, H, W)

			# Global Context extraction
			ctx_vec, c_emb = self.global_extractor(scores_data, misc_data, constrs_site)
			flat_tiles = c_emb.flatten(start_dim=2) # (N, CS, 3*D)
			ctx_expanded = ctx_vec.unsqueeze(1).expand(-1, CONSTR_SITE_SIZE, -1) # (N, CS, C_ctx)

			# =================================================================
			if self.version == 50: # Siamese Spatial Pooling
				# Process P0 and Opponents through SHARED stem
				feat_p0 = self.shared_stem(spatial_p0)
				feat_opp_flat = self.shared_stem(spatial_opp_flat) # (N * num_opp, C_sp, 13, 13)
				
				# Independent Opponent Summary (Pool to remove rotation/position constraints)
				feat_opp = feat_opp_flat.view(N, num_opp, -1, H, W)
				opp_avg = F.adaptive_avg_pool2d(feat_opp.flatten(0, 1), 1).view(N, num_opp, -1).mean(dim=1) # (N, C_sp)
				opp_max = F.adaptive_max_pool2d(feat_opp.flatten(0, 1), 1).view(N, num_opp, -1).max(dim=1)[0] # (N, C_sp)
				opp_summary = torch.cat([opp_avg, opp_max], dim=1) # (N, 2 * C_sp)

				# Continue P0 processing
				feat = self.p0_trunk(feat_p0)
				
				# Build rich context combining explicit global data + implicit opponent summary
				deep_ctx = self.deep_ctx(torch.cat([ctx_vec, opp_summary], dim=1))
				
				# Policy (Late fusion)
				board_features = self.proj_board(feat).permute(0, 2, 3, 1).unsqueeze(1)
				deep_ctx_expanded = deep_ctx.unsqueeze(1).expand(-1, CONSTR_SITE_SIZE, -1)
				fused_tiles = torch.cat([flat_tiles, deep_ctx_expanded], dim=-1)
				
				tile_features = self.proj_tile(fused_tiles)
				orient_features = self.proj_orient(tile_features).view(tile_features.shape[0], tile_features.shape[1], 6, -1)
				
				prod = board_features.unsqueeze(2) * orient_features.unsqueeze(3).unsqueeze(4)
				logits = prod.sum(dim=-1).permute(0, 1, 3, 4, 2)
				pi = torch.where(valid_actions, logits.flatten(1), self.lowvalue)
				
				# Value
				pooled_spatial = F.adaptive_avg_pool2d(feat, 1).flatten(1)
				v = self.val_head(torch.cat([pooled_spatial, deep_ctx], dim=1))

			# =================================================================
			elif self.version == 51: # Opponent Threat Attention
				feat = self.spatial_trunk(spatial_p0)
				deep_ctx = self.deep_ctx(ctx_vec)

				# Process opponents through light stem
				feat_opp_flat = self.opp_stem(spatial_opp_flat) # (N*num_opp, C_opp, 13, 13)
				
				# Attention mechanism: Q(Tiles) -> K,V(Opponents)
				q = self.q_proj(flat_tiles) # (N, CS, C_opp)
				
				k = self.k_proj(feat_opp_flat).view(N, num_opp, -1, H*W).permute(0, 2, 1, 3).reshape(N, -1, num_opp * H * W) # (N, C_opp, num_opp*169)
				v = self.v_proj(feat_opp_flat).view(N, num_opp, -1, H*W).permute(0, 1, 3, 2).reshape(N, num_opp * H * W, -1) # (N, num_opp*169, C_opp)
				
				# Scaled Dot-Product Attention
				C_opp_scale = k.shape[1] ** 0.5
				attn_logits = torch.bmm(q, k) / C_opp_scale # (N, CS, num_opp*169)
				attn_weights = F.softmax(attn_logits, dim=-1)
				threat_vec = torch.bmm(attn_weights, v) # (N, CS, C_opp)

				# Policy (Fuse Tile + Context + Threat Vector)
				board_features = self.proj_board(feat).permute(0, 2, 3, 1).unsqueeze(1)
				deep_ctx_expanded = deep_ctx.unsqueeze(1).expand(-1, CONSTR_SITE_SIZE, -1)
				fused_tiles = torch.cat([flat_tiles, deep_ctx_expanded, threat_vec], dim=-1) # (N, CS, 3*D + C_ctx + C_opp)
				
				tile_features = self.proj_tile(fused_tiles)
				orient_features = self.proj_orient(tile_features).view(tile_features.shape[0], tile_features.shape[1], 6, -1)
				
				prod = board_features.unsqueeze(2) * orient_features.unsqueeze(3).unsqueeze(4)
				logits = prod.sum(dim=-1).permute(0, 1, 3, 4, 2)
				pi = torch.where(valid_actions, logits.flatten(1), self.lowvalue)
				
				# Value
				pooled_spatial = F.adaptive_avg_pool2d(feat, 1).flatten(1)
				v = self.val_head(torch.cat([pooled_spatial, deep_ctx], dim=1))

		elif self.version in [40, 41, 42]:
			# --- COMMON PRE-PROCESSING ---
			# Spatial Embedding
			descr_long = board_descr_p0.clamp(min=0., max=len(CODES_LIST)-1).long()
			spatial_emb = self.embed(descr_long).permute(0, 3, 1, 2) # (N, D, 13, 13)
			spatial_p0 = torch.cat([spatial_emb, board_height_p0], dim=1) # (N, D+1, 13, 13)
			
			# Global Context Processing
			ctx_vec, c_emb = self.global_extractor(scores_data, misc_data, constrs_site) # ctx_vec: (N, C_ctx), c_emb: (N, CS, 3, D)
			flat_tiles = c_emb.flatten(start_dim=2) # (N, CS, 3*D)
			
			# --- ARCHITECTURE SPECIFIC FORWARD ---
			if self.version == 40: # FiLM-Conditioned MobileNet
				feat = self.stem(spatial_p0)
				
				# Block 1 + FiLM
				feat = self.block1(feat)
				g1, b1 = self.film1_gamma(ctx_vec), self.film1_beta(ctx_vec)
				feat = self.film1(feat, g1, b1)
				
				# Block 2 + FiLM
				feat = self.block2(feat)
				g2, b2 = self.film2_gamma(ctx_vec), self.film2_beta(ctx_vec)
				feat = self.film2(feat, g2, b2)
				
				# Policy computation
				board_features = self.proj_board(feat).permute(0, 2, 3, 1).unsqueeze(1) # (N, 1, 13, 13, C_sp)
				tile_features = self.proj_tile(flat_tiles) # (N, CS, C_sp)
				orient_features = self.proj_orient(tile_features).view(tile_features.shape[0], tile_features.shape[1], 6, -1) # (N, CS, 6, C_sp)
				
				prod = board_features.unsqueeze(2) * orient_features.unsqueeze(3).unsqueeze(4) # (N, CS, 6, 13, 13, C_sp)
				logits = prod.sum(dim=-1).permute(0, 1, 3, 4, 2) # (N, CS, 13, 13, 6)
				pi = torch.where(valid_actions, logits.flatten(1), self.lowvalue)
				
				# Value computation
				pooled_spatial = F.adaptive_avg_pool2d(feat, 1).flatten(1) # (N, C_sp)
				v = self.val_head(torch.cat([pooled_spatial, ctx_vec], dim=1))
				
			elif self.version == 41: # Early-Broadcast Bottleneck
				# Broadcast ctx_vec to spatial dimensions
				broadcast_ctx = ctx_vec.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, CITY_SIZE, CITY_SIZE)
				mixed_input = torch.cat([spatial_p0, broadcast_ctx], dim=1) # (N, D+1+C_ctx, 13, 13)
				
				feat = self.bottleneck(mixed_input)
				feat = self.spatial_trunk(feat)
				
				# Policy computation
				board_features = self.proj_board(feat).permute(0, 2, 3, 1).unsqueeze(1)
				tile_features = self.proj_tile(flat_tiles)
				orient_features = self.proj_orient(tile_features).view(tile_features.shape[0], tile_features.shape[1], 6, -1)
				
				prod = board_features.unsqueeze(2) * orient_features.unsqueeze(3).unsqueeze(4)
				logits = prod.sum(dim=-1).permute(0, 1, 3, 4, 2)
				pi = torch.where(valid_actions, logits.flatten(1), self.lowvalue)
				
				# Value computation
				pooled_spatial = F.adaptive_avg_pool2d(feat, 1).flatten(1)
				v = self.val_head(torch.cat([pooled_spatial, ctx_vec], dim=1))

			elif self.version == 42: # Asymmetric Dual-Stream
				# Deep context stream
				deep_ctx = self.deep_ctx(ctx_vec)
				
				# Spatial stream
				feat = self.spatial_trunk(spatial_p0)
				
				# Policy computation (Late fusion of deep_ctx into tile embeddings)
				board_features = self.proj_board(feat).permute(0, 2, 3, 1).unsqueeze(1)
				
				# Inject global understanding into the tiles before projection
				ctx_expanded = deep_ctx.unsqueeze(1).expand(-1, CONSTR_SITE_SIZE, -1) # (N, CS, C_ctx)
				fused_tiles = torch.cat([flat_tiles, ctx_expanded], dim=-1) # (N, CS, 3*D + C_ctx)
				
				tile_features = self.proj_tile(fused_tiles)
				orient_features = self.proj_orient(tile_features).view(tile_features.shape[0], tile_features.shape[1], 6, -1)
				
				prod = board_features.unsqueeze(2) * orient_features.unsqueeze(3).unsqueeze(4)
				logits = prod.sum(dim=-1).permute(0, 1, 3, 4, 2)
				pi = torch.where(valid_actions, logits.flatten(1), self.lowvalue)
				
				# Value computation
				pooled_spatial = F.adaptive_avg_pool2d(feat, 1).flatten(1)
				v = self.val_head(torch.cat([pooled_spatial, deep_ctx], dim=1))

		elif self.version in [1]:
			x_1d = torch.cat([scores_data.flatten(1), constrs_site.flatten(1), globals_data, boards_descr.flatten(1), boards_height.flatten(1)], dim=1) # x_1d.shape = Nx39
			x_1d = F.dropout(self.trunk_1d(x_1d), p=self.args['dropout'], training=self.training) # x_1d.shape = Nx100
			v = self.final_layers_V(x_1d)
			pi = torch.where(valid_actions, self.final_layers_PI(x_1d), self.lowvalue)
		elif self.version in [30, 31, 32]:
			# Work on scores and glob data first
			s1 = F.dropout(self.dense_scores(scores_data.flatten(1)), p=self.args['dropout'], training=self.training) # N,3*NUM_PLAYERS,N_COLORS -> N,S
			g1 = F.dropout(self.dense_globs(globals_data), p=self.args['dropout'], training=self.training) # 2 -> N,G

			# Convert boards_descr to embeddings (need to clamp when input is random)
			boards_descr_long = boards_descr.clamp(min=0., max=len(CODES_LIST)-1).long()
			boards_descr_embed = self.embed(boards_descr_long) # N,2,12,12,D
			boards = torch.cat([boards_descr_embed, boards_height.unsqueeze(-1), boards_tileID.unsqueeze(-1)], dim=-1) # N,2,12,12,D+2
			boards_4d = [boards[:,i,:,:,:].permute(0,3,1,2) for i in range(self.num_players)] # N,D+2,12,12 (num_players times)
			boards_4d = [self.conv2d_boards(boards_4d[i]) for i in range(self.num_players)] # N,B,12,12 (num_players times)

			# Merge boards + scores + global data to a 4D vector
			scores_4d = s1.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, CITY_SIZE, CITY_SIZE) # N, S, 12, 12
			glob_4d   = g1.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, CITY_SIZE, CITY_SIZE) # N, G, 12, 12
			fused_4d = torch.cat(boards_4d + [scores_4d, glob_4d], dim=1) # N,self.num_players*B+G+S,12,12

			# Convert constrs_site to embeddings (need to clamp when input is random)
			constrs_long = constrs_site.clamp(min=0., max=len(CODES_LIST)-1).long()
			constrs_embed = self.embed(constrs_long) # N,CS,3,D
			constrs_3d = constrs_embed.flatten(start_dim=0, end_dim=1).permute(0, 2, 1) # N*CS, D, 3
			constrs_3d = F.dropout(self.conv1d_constr(constrs_3d), p=self.args['dropout'], training=self.training) # N*CS, T
			constrs_3d = constrs_3d.squeeze(-1).view(constrs_embed.shape[0], constrs_embed.shape[1], -1) # N, CS, T

			# Merge constrs + scores + global data to a 3D vector
			scores_3d = s1.unsqueeze(1).expand(-1, CONSTR_SITE_SIZE, -1) # N, CS, S
			glob_3d = g1.unsqueeze(1).expand(-1, CONSTR_SITE_SIZE, -1) # N, CS, G
			fused_3d = torch.cat([constrs_3d, scores_3d, glob_3d], dim=-1) # N,CS,T+G+S

			# Compute policy
			fused_3d_proj = self.proj_i(fused_3d) # N, CS, r
			fused_3d_proj = self.b1n(fused_3d_proj.permute(0,2,1)).permute(0,2,1)
			fused_3d_proj = fused_3d_proj.unsqueeze(2).unsqueeze(2) # N, CS, 1, 1, r
			fused_4d_proj = self.proj_p(fused_4d) # N, r, 12, 12
			fused_4d_proj = fused_4d_proj.permute(0, 2, 3, 1).unsqueeze(1) # N, 1, 12, 12, r	
			if self.version == 30:
				logits = torch.einsum('nchwt,ot->nchwo', fused_3d_proj*fused_4d_proj, self.U_o) # N, CS, 12, 12, r
			else:
				h_o = self.proj_o(fused_3d)                   # → (N, CS, 6*r)
				h_o = h_o.view(h_o.shape[0], h_o.shape[1], 6, -1).unsqueeze(3).unsqueeze(4) # → (N, CS, 6, 1, 1, r)
				prod = (fused_3d_proj.unsqueeze(2))*(fused_4d_proj.unsqueeze(1))*h_o        # prod : (N, CS, 6, H, W, r)
				logits = prod.sum(dim=-1).permute(0,1,3,4,2)  # (N,CS,H,W,6)
			pi = torch.where(valid_actions, logits.flatten(1), self.lowvalue)

			# Compute value
			v = self.final_layers_V(fused_3d)
		else:
			raise Exception(f'Unsupported NN version {self.version}')

		return F.log_softmax(pi, dim=1), torch.tanh(v)

