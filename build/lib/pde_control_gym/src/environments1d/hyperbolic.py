import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Optional

from pde_control_gym.src.environments1d.base_env_1d import PDEEnv1D

class TransportPDE1D(PDEEnv1D):
    r""" 
    Transport PDE 1D

    This class implements the 1D Transport PDE and inherits from the class :class:`PDEEnv1D`. Thus, for a full list of of arguments (参数), first see the class :class:`PDEEnv1D` in conjunction (联合) with the arguments presented here
    # 定义传感器噪声函数
    :param sensing_noise_func: Takes in a function that can add sensing noise into the system. Must return the same sensing vector as given as a parameter.
    # 定义初始条件函数 (使用在reset方法中)
    :param reset_init_condition_func: Takes in a function used during the reset method for setting the initial PDE condition :math:`u(x, 0)`.
    # 定义反应系数函数 (使用在reset方法中)
    :param reset_recirculation_func: Takes in a function used during the reset method for setting the initial plant parameter :math:`\beta` vector at the start of each epsiode.
    # 定义传感器位置
    :param sensing_loc: Sets the sensing location as either ``"full"``, ``"collocated"``, or ``"opposite"`` which indicates whether the full state, the boundary at the same side of the control, or boundary at the opposite side of control is given as the observation at each time step.
    # 定义控制器如何作用于边界 (直接控制边界or边界导数) 
    :param control_type: The control location can either be given as a ``"Dirchilet"`` or ``"Neumann"`` boundary conditions and is always at the ``X`` point. 
    # 仅在传感器与控制器在不同位置时使用,指定感知的边界条件类型,感知点位置位于0点。
    :param sensing_type: Only used when ``sensing_loc`` is set to ``opposite``. In this case, the sensing can be either given as ``"Dirchilet"`` or ``"Neumann"`` and is given at the ``0`` point.
    # 限制 PDE 状态大小,一个布尔值用于决定是否在PDE状态的L^2范数超过max_state_value时提前终止该 episode。
    :param limit_pde_state_size: This is a boolean which will terminate the episode early if :math:`\|u(x, t)\|_{L_2} \geq` ``max_state_value``.
    # 仅在 limit_pde_state_size 为 True 时使用。
    :param max_state_value: Only used when ``limit_pde_state_size`` is ``True``. Then, this sets the value for which the :math:`L_2` norm of the PDE will be compared to at each step asin ``limit_pde_state_size``.
    # 设置最大控制值,在normalize = True 时使用,将动作空间映射到[-max_control_value, max_control_value]。
    :param max_control_value: Sets the maximum control value input as between [``-max_control_value``, ``max_control_value``] and is used in the normalization of action inputs.
    # 设置控制器采样率,控制器应用于 PDE 的采样率。允许 PDE 在更高分辨率下模拟，而控制器以较低频率作用。
    # 让 PDE 用高频模拟、控制器用低频施加动作，是一种非常有效的结构分离（control-physics decoupling）。
    :param control_sample_rate: Sets the sample rate at which the controller is applied to the PDE. This allows the PDE to be simulated at a smaller resolution then the controller.
    """
    def __init__(self, sensing_noise_func: Callable[[np.ndarray], np.ndarray],
                 reset_init_condition_func: Callable[[int], np.ndarray],
                 reset_recirculation_func: Callable[[int], np.ndarray], 
                 sensing_loc: str = "full",
                 control_type: str= "Dirchilet", 
                 sensing_type: str = "Dirchilet", 
                 limit_pde_state_size: bool = False, 
                 max_state_value: float = 1e10, 
                 max_control_value: float = 20, 
                 control_sample_rate: float=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        # config（字典）
        # ↓
        # ChildClass(**config)
        # ↓
        # ChildClass.__init__(self, ..., **kwargs) # kwargs是一个字典，包含了除ChildClass.__init__()方法中定义的参数以外的其他参数
        # ↓
        # super().__init__(**kwargs) # 调用父类的构造函数，将该类的（**kwargs）传递给父类，
        # ↓
        # ParentClass.__init__(self, ...) # 生成的数据存在self（self只有一个，父类和子类共享）中，直接访问
        # ↓
        # 初始化父类的数据/结构
        # ↓
        # 返回子类继续初始化自己的参数
        
        self.sensing_noise_func = sensing_noise_func
        self.reset_init_condition_func = reset_init_condition_func 
        self.reset_recirculation_func = reset_recirculation_func
        self.sensing_loc = sensing_loc
        self.control_type = control_type
        self.sensing_type = sensing_type
        self.limit_pde_state_size = limit_pde_state_size
        self.max_state_value = max_state_value
        self.max_control_value = max_control_value
        self.control_sample_rate = control_sample_rate
	    # Observation space changes depending on sensing
        match self.sensing_loc:
            case "full":
                self.observation_space = spaces.Box(
                    np.full(self.nx, -self.max_state_value, dtype="float32"),
                    np.full(self.nx, self.max_state_value, dtype="float32"),
                )
            case "collocated" | "opposite":
                self.observation_space = spaces.Box(
                    np.full(1, -self.max_state_value, dtype="float32"),
                    np.full(1, self.max_state_value, dtype="float32"),
                )
            case _:
                raise Exception(
                    "Invalid sensing_loc parameter. Please use 'full', 'collocated', or 'opposite'. See documentation for details."
                )

        # Setup configurations for control and sensing. Messy, but done once, explicitly before runtime to setup return and control functions
        # There is a trick here where noise is a function call itself. Important that noise is a single argument function that returns a single argument
        match self.control_type:
            case "Neumann":
                self.control_update = lambda control, state, dx: control * dx + state
                match self.sensing_loc:
                    # Neumann control u_x(1), full state measurement
                    case "full":
                        self.sensing_update = lambda state, dx, noise: noise(state)
                    # Neumann control u_x(1), Dirchilet sensing u(1)
                    case "collocated":
                        self.sensing_update = lambda state, dx, noise: noise(state[-1])
                    case "opposite":
                        match self.sensing_type:
                            # Neumann control u_x(1), Neumann sensing u_x(0)
                            case "Neumann":
                                self.sensing_update = lambda state, dx, noise: noise(
                                    (state[1] - state[0]) / dx
                                )
                            # Neumann control u_x(1), Dirchilet sensing u(0)
                            case "Dirchilet":
                                self.sensing_update = lambda state, dx, noise: noise(state[0])
                            case _:
                                raise Exception(
                                    "Invalid sensing_type parameter. Please use 'Neumann' or 'Dirchilet'. See documentation for details."
                                )
                    case _:
                        raise Exception(
                            "Invalid sensing_loc parameter. Please use 'full', 'collocated', or 'opposite'. See documentation for details."
                        )
            case "Dirchilet":
                self.control_update = lambda control, state, dt: control
                match self.sensing_loc:
                    # Dichilet control u(1), full state measurement
                    case "full":
                        self.sensing_update = lambda state, dx, noise: noise(state)
                    # Dichilet control u(1), Neumann sensing u_x(1)
                    case "collocated":
                        self.sensing_update = lambda state, dx, noise: noise(
                            (state[-1] - state[-2]) / dx
                        )
                    case "opposite":
                        match self.sensing_type:
                            # Dichilet control u(1), Neumann sensing u_x(0)
                            case "Neumann":
                                self.sensing_update = lambda state, dx, noise: noise(
                                    (state[1] - state[0]) / dx
                                )
                            # Dirchilet control u(1), Dirchilet sensing u(0)
                            case "Dirchilet":
                                self.sensing_update = lambda state, dx, noise: noise(
                                    state[0]
                                )
                            case _:
                                raise Exception(
                                    "Invalid sensing_type parameter. Please use 'Neumann' or 'Dirchilet'. See documentation for details."
                                )
            case _:
                raise Exception(
                    "Invalid control_type parameter. Please use 'Neumann' or 'Dirchilet'. See documentation for details."
                )

    def step(self, control: float):
        """
        step

        Moves the PDE with control action forward ``control_sample_rate*dt`` steps.

        :param control: The control input to apply to the PDE at the boundary.
        """
        Nx = self.nx
        dx = self.dx
        dt = self.dt
        sample_rate = int(round(self.control_sample_rate/dt))
        i = 0
        # Actions are applied at a slower rate then the PDE is simulated at
        while i < sample_rate and self.time_index < self.nt-1:
            self.time_index += 1
            # Explicit update of u according to finite difference derivation
            self.u[self.time_index][-1] = self.normalize(self.control_update(
                control, self.u[self.time_index][-2], dx), self.max_control_value
            )
            self.u[self.time_index][0 : Nx - 1] = self.u[self.time_index - 1][
                0 : Nx - 1
            ] + dt * (
                (
                    self.u[self.time_index - 1][1:Nx]
                    - self.u[self.time_index - 1][0 : Nx - 1]
                )
                / dx
                + (self.u[self.time_index - 1][0] * self.beta)[0 : Nx - 1]
            )
            i += 1
        terminate = self.terminate()
        truncate = self.truncate()
        return (
            self.sensing_update(
                self.u[self.time_index],
                self.dx,
                self.sensing_noise_func,
            ),
            self.reward_class.reward(self.u, self.time_index, terminate, truncate, self.u[self.time_index][-1]),
            terminate,
            truncate, 
            {},
        )

    def terminate(self):
        """
        terminate

        Determines whether the episode should end if the ``T`` timesteps are reached
        """
        if self.time_index >= self.nt - 1:
            return True
        else:
            return False

    def truncate(self):
        """
        truncate 

        Determines whether to truncate the episode based on the PDE state size and the vairable ``limit_pde_state_size`` given in the PDE environment intialization.
        """
        if (
            self.limit_pde_state_size
            and np.linalg.norm(self.u[self.time_index], 2)  >= self.max_state_value
        ):
            return True
        else:
            return False
         

    # Resets the system state
    def reset(self, seed: Optional[int]=None, options: Optional[dict]=None):
        """
        reset 

        :param seed: Allows a seed for initialization of the envioronment to be set for RL algorithms.
        :param options: Allows a set of options for the initialization of the environment to be set for RL algorithms.

        Resets the PDE at the start of each environment according to the parameters given during the PDE environment intialization
        """
        try:
            init_condition = self.reset_init_condition_func(self.nx)
            beta = self.reset_recirculation_func(self.nx)
        except:
            raise Exception(
                "Please pass both an initial condition and a recirculation function in the parameters dictionary. See documentation for more details"
                )
        self.u = np.zeros(
            (self.nt, self.nx), dtype=np.float32
        )
        self.u[0] = init_condition
        self.time_index = 0
        self.beta = beta
        return (
            self.sensing_update(
                self.u[self.time_index],
                self.dx,
                self.sensing_noise_func,
            ),
            {},
        )
