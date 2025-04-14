import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod
from typing import Type
from pde_control_gym.src.rewards import BaseReward

# 继承自gym.env,说明是一个Gym兼容的环境
class PDEEnv1D(gym.Env):
# 定义了一维PDE控制问题的抽象基类，便于之后使用
    """
    This is the base env for all 1D PDE problems. All 1D custom environments should inherit this environment and implement the according methods
    :param T: The end time of the simulation.
    :param dt: The temporal timestep of the simulation.
    :param X: The spatial length of the simulation.
    :param dx: The spatial timestep of the simulation.
    :param reward_class: An instance of the reward class to specify user reward for each simulation step. Must inherit BaseReward class. See `reward documentation <../../utils/rewards.html>`_ for detials.
    :param normalize: Chooses whether to take action inputs between -1 and 1 and normalize them to betwen (``-max_control_value``, ``max_control_value``) or to leave inputs unaltered（保持输入不变）. ``max_control_value`` is environment specific so please see the environment for details. 
    """
    def __init__(self, T: float, dt: float, X: float, dx: float, reward_class: Type[BaseReward], normalize: bool = False):
        # 调用父类的构造函数，将该类的self对象传递给父类
        # super().__init__()是一个特殊的方法，用于调用父类的构造函数；是所有 Gym 环境的标准写法，确保环境可以被 gym 注册、创建和调用
        super(PDEEnv1D, self).__init__()
        # Build parameters for number of time steps and number of spatial steps
        self.nt = int(round(T/dt)) # 时间离散步数
        self.nx = int(round(X/dx)) # 空间离散步数
        self.dt = dt 
        self.T = T
        self.dx = dx
        self.X = X

      	# Observation Space handled in individual environment due to sensing modifications
        # 在强化学习中，观察空间（Observation Space）定义了环境可以提供给智能体的状态信息的范围和结构。
        # 它通常包括环境的状态变量，例如位置、速度或其他与问题相关的特征。观察空间的处理在每个具体的环境中完成。

        # Action space is always just boundary control. Normalized to -1 and 1 but gets expanded according to max_control_value
        # # 定义动作空间（Action Space），表示智能体可以采取的动作范围和结构。一维实数范围是[-1,1]
        # self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype="float32") 
        self.action_space = spaces.Box(
            np.full(1, -1, dtype="float32"), np.full(1, 1, dtype="float32")
        )
        if normalize:
            # 需要映射：[-1, 1] -> [-max, max] 一般情况写法： u = (a + 1) * max - max
            self.normalize = lambda action, max_value : (action + 1)*max_value - max_value
            # lambda 定义一个匿名函数，接受两个参数action和max_value，并返回一个值
            # def som_func(action, max_value):
            #     return (action + 1) * max_value - max_value
            # self.normalize = som_func
        else:
            self.normalize = lambda action, max_value : action
        # Holds entire system state. Modified with ghost points for parabolic PDE
        self.u = np.zeros((self.nt, self.nx)) 

        self.time_index = 0 # 时间索引

        # Setup reward function.
        self.reward_class = reward_class

    # 定义了一个抽象方法，表示该类是一个抽象基类，不能直接实例化，必须要撰写函数
    @abstractmethod 
    def step(self, action: np.ndarray):
        """
        step

        Implements the environment behavior for a single timestep depending on a given action

        :param action: The action to take in the environment. 
        """
        # pass 用来占位，子类的时候填写
        pass

    @abstractmethod
    def reset(self, init_cond: np.ndarray, recirculation_func):
        """
        reset 

        Resets the environment at the start of each epsiode

        :param init_cond: The intial condition to reset the PDE :math:`u(x, 0)` to. 

        :param recirculation_func: Specifies the plant parameter function. See each individual environment for details on implementation.
        """
        pass
