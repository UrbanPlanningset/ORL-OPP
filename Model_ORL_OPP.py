import math
import random
from collections import defaultdict
from copy import deepcopy
from gym import spaces
import numpy as np
import random
from gym import spaces
import torch
import torch.nn as nn
from tqdm import tqdm
from termcolor import cprint
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from haversine import haversine
import geohash2 as geohash
from tqdm import tqdm
from termcolor import cprint
from models_general import GNN,MLP


class Model(nn.Module):
    def __init__(self, num_nodes, graph=None, device="cpu", args=None, embeddings=None, mapping=None, traffic_matrix=None):
        super(Model, self).__init__()
        self.args = args
        if embeddings is None:
            self.embeddings = nn.Embedding(num_nodes, args.embedding_size)
            self.embeddings = nn.Embedding.from_pretrained(self.embeddings.weight, freeze=not args.trainable_embeddings)
        else:
            self.embeddings = nn.Embedding.from_pretrained(embeddings.float(), freeze=not args.trainable_embeddings)

        if (args.gnn is not None):
            cprint("GNN: {}".format(args.gnn), "cyan")
            node_feature_size_for_gnn = self.embeddings.weight.shape[1]
            self.GNN = GNN(
                node_feature_size=node_feature_size_for_gnn,
                output_embedding_size=args.embedding_size,
                num_layers=args.gnn_layers,
                hidden_dim=args.hidden_size,
                graph=graph,
                gnn_type=args.gnn
            )

        input_size = 6 * args.embedding_size
        self.mapping = mapping
        self.device = device

        if args.traffic:
            input_size = 8 * args.embedding_size  # one for traffic
            self.traffic_matrix = nn.Embedding.from_pretrained(traffic_matrix, freeze=True)
            self.traffic_linear_initial = nn.Linear(self.traffic_matrix.weight.shape[1], 2 * args.embedding_size)
        if args.attention:
            self.self_attention = nn.MultiheadAttention(2 * self.embeddings.weight.shape[1], args.num_heads)

        self.confidence_model = MLP(input_dim=input_size, output_dim=1, num_layers=args.num_layers, hidden_dim=args.hidden_size)

        self.build_model = None

    def _build_model(self, embed_dim):
        return nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def _get_vec(self, source):
        device = self.device
        edge_to_node_mapping = self.mapping

        source_left = edge_to_node_mapping[source][0]
        source_right = edge_to_node_mapping[source][1]

        if self.args.gnn is not None:
            self.GNN.data.x = self.embeddings.weight
            embeddings = self.GNN()
        else:
            embeddings = self.embeddings.weight

        embeddings = torch.cat((torch.zeros(embeddings.shape[1]).reshape(1, -1).to(self.device), embeddings), dim=0)

        vec_left = embeddings[1 + source_left].unsqueeze(0)
        vec_right = embeddings[1 + source_right].unsqueeze(0)
        vec = torch.cat((vec_left, vec_right), dim=1)
        return vec

    def forward(self, input, return_vec=False):
        if return_vec:
            return self._get_vec(input)
        else:

            if self.build_model is None:
                embed_dim = input.shape[-1]  #
                self.build_model = self._build_model(embed_dim).to(self.device)
            return self.build_model(input)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []

    def add(self, experience, td_error):
        self.buffer.append(experience)
        self.priorities.append((abs(td_error) + 1e-6) ** self.alpha)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)

    def sample(self, batch_size):
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        return samples, indices

    def update_priorities(self, indices, new_td_errors):
        for idx, td in zip(indices, new_td_errors):
            self.priorities[idx] = (abs(td) + 1e-6) ** self.alpha