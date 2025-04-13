from pde_control_gym.src.rewards.base_reward import BaseReward
from typing import Optional
import numpy as np

class TunedReward1D(BaseReward):
    """
    TunedReward1D

    This is a custom reward used for successfully implementing the model-free boundary controller for the 1D environments as given in the `paper <https://google.com>`_ documenting the benchmark. 

    :param nt: The number of maximum timesteps for the episode simulation. No default: Error is thrown if not specified.
    :param truncate_penalty: Allows the user to specify a penalty reward for each remaining timestep in case the episode is ended early. Default is :math:`-1e4`.
    :param terminate_reward: Allows the user to add a reward for reaching the full length of the episode: Default is :math:`1e2`.

    """
    
    def __init__(self, nt: int, truncate_penalty: float = -1e-4, terminate_reward: float = 1e2):
        # Exception Handling
        if nt is None:
            raise Exception("Number of simulation steps must be specified in the TunedReward1D class.")
        self.nt = nt
        self.truncate_penalty = truncate_penalty
        self.terminate_reward = terminate_reward

    def reward(self, uVec: np.ndarray =None, time_index: int = None, terminate: Optional[bool] =None, truncate: Optional[bool] =None, action: Optional[float] =None):
        r""" 
        reward
        # 必须参数，表示PDE的解向量，用于计算奖励
        :param uVec: (required) This is the solution vector of the PDE of which to compute the reward on.
        # 表示当前步的索引
        :param time_index: (required) This is the time at which to compute the reward. (Given in terms of index of uVec).
        # 表示当前是否是终止状态
        :param terminate: States whether the episode is the terminal episode.
        # 表示当前是否是截断状态
        :param truncate: States whether the epsiode is truncated, or ending early.
        :param action: Ignored in this reward - needed to inherit from base reward class.
        """
        # 根据 PDE 解的状态、时间步索引以及是否终止或截断，计算奖励值。
        # 如果 terminate 为真，且当前步的解向量的范数小于20触发
        if terminate and np.linalg.norm(uVec[time_index]) < 20:
            return (self.terminate_reward - np.sum(abs(uVec[:, -1]))/1000 - np.linalg.norm(uVec[time_index]))
        # 如果 truncate 为真，返回剩余步数的惩罚
        if truncate:
            return self.truncate_penalty*(self.nt-time_index)
        return np.linalg.norm(uVec[time_index-100])-np.linalg.norm(uVec[time_index])
