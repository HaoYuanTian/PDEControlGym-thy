from pde_control_gym.src.environments1d import TransportPDE1D, ReactionDiffusionPDE1D
from pde_control_gym.src.environments2d import NavierStokes2D
from pde_control_gym.src.rewards import BaseReward, NormReward, TunedReward1D, NSReward

__all__ = ["TransportPDE1D", "ReactionDiffusionPDE1D", "NavierStokes2D", "BaseReward", "NormReward", "TunedReward1D", "NSReward"]

# 在包目录下有一个空的 __init__.py 文件就像一个“标签”，告诉 Python：
# “我是个包，可以被 import，用我吧～”
# 它虽然什么都不做，但它的存在 本身就有意义。
# 但也可以进行模块整合和接口管理。
# 功能一：导入关键类以统一出口。让这些类在pde_control_gym.src这个模块级别就可以直接访问；减少用户记住内部结构的负担。
# 用 from pde_control_gym.src import TransportPDE1D 代替：
#    from pde_control_gym.src.environments1d import TransportPDE1D
# 功能二：定义模块导出接口
# 当使用 from pde_control_gym.src import * 时只导入__all__中列出的类
# 这个 __init__.py 文件是为了让 pde_control_gym.src 这个模块：
# •	可以作为整洁的统一入口；
# •	让用户更方便导入环境类和奖励函数类；
# •	并限制导入行为，让 API 更干净清晰。