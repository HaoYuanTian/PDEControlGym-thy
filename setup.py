from setuptools import setup

setup(
    
    name="pdecontrolgym", # ✅ pip 包名，用于安装管理，在 pip list中显示
    version="0.0.1", 
    install_requires=["gymnasium", 
            "numpy", 
            "matplotlib"], 
        )

# 将你当前目录下的项目作为一个 “可编辑的开发包” 安装进 Python 环境中。
# 只有包含_init__.py文件的目录才会被视为一个Python包，然后所有的包会被打包为一个包，命名为 $name$ ，在 pip list 中显示。
# PDEControlGym/                  ← 项目根目录
# ├── setup.py
# ├── pde_control_gym/           ← ✅ 模块的“根目录”，这里必须有 __init__.py
# │   ├── __init__.py
# │   └── src/
# │       ├── __init__.py
# │       └── environments1d.py
# 在这种情况下 src 是 pde_control_gym 的子模块 可以 import pde_control_gym.src
# project/
# ├── setup.py               ← name="pdecontrolgym"
# ├── pde_control_gym/       ← ✅ 模块 1
# │   └── __init__.py
# ├── utils/                 ← ✅ 模块 2
# │   └── __init__.py
# 在这种情况下 import pde_control_gym
#              import utils           是两个不同的模块。
# 因为使用pip install -e .（可编辑安装），指向了项目的源码目录，让解释器直接从源码读取模块，而不是依赖于packages= find_packages()中的打包结构。