from pde_control_gym.src.rewards.norm_reward import NormReward
from pde_control_gym.src.rewards.tuned_reward_1d import TunedReward1D
from pde_control_gym.src.rewards.base_reward import BaseReward
from pde_control_gym.src.rewards.ns_reward import NSReward

__all__ = ["NormReward", "TunedReward1D", "BaseReward", "NSReward"]

# 当使用from pde_control_gym.src.rewards import *
# 会导入在__all__中列入的4个对象"NormReward", "TunedReward1D", "BaseReward", "NSReward"。
# 方便打包管理。