import gymnasium as gym
import numpy as np
import math
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
import pde_control_gym
from pde_control_gym.src import TunedReward1D
from examples.transportPDE.DeepONet import CustomFeatureExtractor
from torch.utils.tensorboard import SummaryWriter



# THIS EXAMPLE TRAINS A SAC AGENT FOR THE HYPERBOLIC PDE PROBLEM. 
# The model is saved every 10k timesteps to the directory ./logsSAC/
# The tensorboard results are saved to the directory
# ./tb/

# NO NOISE
def noiseFunc(state):
    return state

# Chebyshev Polynomial Beta Functions
def solveBetaFunction(x, gamma):
    beta = np.zeros(len(x), dtype=np.float32)
    for idx, val in enumerate(x):
        beta[idx] = 5*math.cos(gamma*math.acos(val))
    return beta

# Kernel function solver for backstepping
def solveKernelFunction(theta):
    kappa = np.zeros(len(theta))
    for i in range(0, len(theta)):
        kernelIntegral = 0
        for j in range(0, i):
            kernelIntegral += (kappa[i-j]*theta[j])*dx
        kappa[i] = kernelIntegral  - theta[i]
    return np.flip(kappa)

# Control convolution solver
def solveControl(kernel, u):
    res = 0
    for i in range(len(u)):
        res += kernel[i]*u[i]
    return res*1e-2

# Set initial condition function here
def getInitialCondition(nx):
    return np.ones(nx)*np.random.uniform(1, 10)

# Returns beta functions passed into PDE environment. Currently gamma is always
# set to 7.35, but this can be modified for further problesms
def getBetaFunction(nx):
    return solveBetaFunction(np.linspace(0, 1, nx), 7.35)

# Timestep and spatial step for PDE Solver
T = 5
dt = 1e-4
dx = 1e-2
X = 1

hyperbolicParameters = {
        "T": T, 
        "dt": dt, 
        "X": X,
        "dx": dx, 
        "reward_class": TunedReward1D(int(round(T/dt)), -1e3, 3e2),
        "normalize":True, 
        "sensing_loc": "full", 
        "control_type": "Dirchilet", 
        "sensing_type": None,
        "sensing_noise_func": lambda state: state,
        "limit_pde_state_size": True,
        "max_state_value": 1e10,
        "max_control_value": 20,
        "reset_init_condition_func": getInitialCondition,
        "reset_recirculation_func": getBetaFunction,
        "control_sample_rate": 0.1,
}

# Make the hyperbolic PDE gym
env = gym.make("PDEControlGym-TransportPDE1D", **hyperbolicParameters)

# 定义策略参数
policy_kwargs = dict(
    features_extractor_class=CustomFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=101),
)

model = SAC("MlpPolicy", 
            env, 
            verbose=1,
            policy_kwargs=policy_kwargs,
            learning_rate = 3e-4,
            tensorboard_log="./PDEControlGym/tb2/")
# # 打印网络结构
# print(model.policy)

# Train for 1 Million timesteps
isTrain = True
if isTrain:
    model.learn(total_timesteps=100_000, tb_log_name="tpde_ONet",)
    model.save("./PDEControlGym/Model/trans_sac_onet.zip")
model = SAC.load("./PDEControlGym/Model/trans_sac_onet.zip")

uStorage = []
rewStorage = []

hyperbolicParameters["reset_init_condition_func"] = lambda nx: np.ones(nx) * 6
env = gym.make("PDEControlGym-TransportPDE1D", **hyperbolicParameters)
obs,__ = env.reset()
uStorage.append(obs)

terminate = False
truncate = False
while not truncate and not terminate:
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, terminate, truncate, info = env.step(action)
    uStorage.append(obs)
    rewStorage.append(rewards)
u = np.array(uStorage)
rewArray = np.array(rewStorage)
print("Total Reward",  np.sum(rewArray))
if truncate:
    print("Truncated")

"""*************************绘图部分*********************"""

# 波形图
res = 1
fig = plt.figure()
spatial = np.linspace(dx, X, int(round(X/dx)) + 1)
temporal = np.linspace(0, T, len(uStorage))
u = np.array(uStorage)

subfigs = fig.subfigures(nrows=1, ncols=1, hspace=0)

subfig = subfigs
subfig.subplots_adjust(left=0.07, bottom=0, right=1, top=1.1)
axes = subfig.subplots(nrows=1, ncols=1, subplot_kw={"projection": "3d", "computed_zorder": False})

for axis in [axes.xaxis, axes.yaxis, axes.zaxis]:
    axis._axinfo['axisline']['linewidth'] = 1
    axis._axinfo['axisline']['color'] = "b"
    axis._axinfo['grid']['linewidth'] = 0.2
    axis._axinfo['grid']['linestyle'] = "--"
    axis._axinfo['grid']['color'] = "#d1d1d1"
    axis.set_pane_color((1,1,1))
    
meshx, mesht = np.meshgrid(spatial, temporal)
axes.plot_surface(meshx, mesht, u, edgecolor="black",lw=0.2, rstride=2, cstride=15, 
                        alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
axes.plot(np.zeros(len(temporal)), temporal, u[:, 0], color="red", lw=2, antialiased=True)
axes.view_init(10, 15)
axes.invert_xaxis()
axes.set_xlabel(r"x", fontsize=14)
axes.set_ylabel(r"$t(\text{sec})$", fontsize=14)
axes.set_zlabel(r"$v(x, t)$", rotation=90, fontsize=14)
axes.zaxis.set_rotate_label(False)
axes.set_xticks([0, 0.5, 1])
axes.tick_params(axis='both', labelsize=12, pad=1) 
plt.savefig('./PDEControlGym/pics/compare/transportPDE/ONet.png', dpi=300)


