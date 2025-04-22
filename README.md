# 坦克世界大战游戏

这是一个支持AI对战和人类玩家参与的坦克大战游戏，专为多智能体强化学习训练设计。

## 项目结构

```
./
├── game/                  # 游戏核心代码
│   ├── entities/          # 游戏实体（坦克、子弹等）
│   ├── map/               # 地图生成和管理
│   └── ui/                # 游戏界面
├── ai/                    # AI相关代码
│   ├── environment.py     # RL环境定义
│   ├── agent.py           # 智能体实现
│   └── training.py        # 训练流程
├── data/                  # 数据收集和存储
│   └── collector.py       # 数据收集器
└── assets/                # 游戏资源（图像、音效等）
```

## 功能特性

- 双人同屏本地对战，支持AI vs AI、AI vs 人类、人类 vs 人类
- 多种类型坦克，参数随机化
- 可破坏和不可破坏地形的随机地图
- 数据收集系统，记录游戏数据用于RL训练
- 强化学习接口，兼容主流RL框架

## 开发计划

1. 实现游戏核心系统（坦克、地图、碰撞检测等）
2. 开发基础AI行为（追踪、攻击等）
3. 设计RL环境和接口
4. 实现数据收集系统
5. 开发和训练RL智能体

## 技术栈

- Python 3.8+
- Pygame (游戏引擎)
- NumPy (数值计算)
- PyTorch/TensorFlow (RL训练)
- Gymnasium (RL环境接口)

## 快速开始

1. 创建conda环境并激活:
```bash
conda create -n tank_game python=3.10
conda activate tank_game
```

2. 安装依赖:
```bash
pip install pygame numpy gymnasium
# 根据需求选择安装PyTorch或TensorFlow
pip install torch  # 或 tensorflow
```

3. 运行游戏:
```bash
python main.py  # 假设入口文件是main.py
```