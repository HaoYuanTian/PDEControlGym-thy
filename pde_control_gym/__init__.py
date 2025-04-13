from gymnasium.envs.registration import register

register(
    id="PDEControlGym-TransportPDE1D", entry_point="pde_control_gym.src:TransportPDE1D"
)

register(
    id="PDEControlGym-ReactionDiffusionPDE1D", entry_point="pde_control_gym.src:ReactionDiffusionPDE1D"
)

register(
    id="PDEControlGym-NavierStokes2D", entry_point="pde_control_gym.src:NavierStokes2D"
)

print("envs initialized")

# register(...) 是一个函数调用。id为这个环境定义的唯一标识符，必须是字符串。之后可以用gym.make("id", **参数字典)来创建强化学习环境
# entry_point指定环境类的位置，格式是“模块路径：类名”，pde_control_gym.src:TransportPDE1D 表示 pde_control_gym.src 模块中，有一个TransportPDE1D。