import deepxde as dde
import torch
import os
import numpy as np
from torch import nn
import torch.nn.functional as F
from gymnasium import spaces
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.torch_layers import create_mlp
import torch.optim as optim


grid = np.linspace(0, 1, 101, dtype=np.float32).reshape(101, 1)
grid = torch.from_numpy(np.array(grid)).cuda()

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=21):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        # 定义你的网络结构
        self.net1 = dde.nn.DeepONetCartesianProd([101, 256], [1, 256], "relu", "Glorot normal").cuda()
        # self.net1.load_state_dict(torch.load("./PDEControlGym/Model/pretrain_deeponet.zip", weights_only=True))
    def forward(self, observations):
        # 前向传播
        x = self.net1((observations, grid))
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(x)
        return x
    
class FNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=21):
        super(FNNFeatureExtractor, self).__init__(observation_space, features_dim)
        # 定义你的网络结构
        self.net1 = torch.nn.Linear(101, 256)
        self.relu = torch.nn.ReLU()
        
    def forward(self, observations):
        # 前向传播
        x = self.net1(observations)
        x = torch.relu(x)

        return x