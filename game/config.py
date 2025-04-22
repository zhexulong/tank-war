# 游戏配置文件

# 屏幕设置
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
TITLE = "坦克世界大战"
FPS = 60

# 坦克设置
TANK_TYPES = {
    "light": {
        "speed_range": (3, 5),
        "health_range": (50, 100),
        "damage_range": (10, 20),
        "reload_time_range": (0.5, 1.5),
        "color": (0, 255, 0)  # 绿色
    },
    "medium": {
        "speed_range": (2, 4),
        "health_range": (100, 150),
        "damage_range": (20, 30),
        "reload_time_range": (1.0, 2.0),
        "color": (255, 255, 0)  # 黄色
    },
    "heavy": {
        "speed_range": (1, 3),
        "health_range": (150, 200),
        "damage_range": (30, 40),
        "reload_time_range": (1.5, 2.5),
        "color": (255, 0, 0)  # 红色
    }
}

# 地图设置
TILE_SIZE = 40
MAP_WIDTH = SCREEN_WIDTH // TILE_SIZE
MAP_HEIGHT = SCREEN_HEIGHT // TILE_SIZE

# 地形类型
TERRAIN_TYPES = {
    "empty": 0,
    "brick": 1,  # 可破坏
    "steel": 2,  # 不可破坏
    "water": 3,  # 不可通过但可射击
    "grass": 4   # 可通过且可隐藏
}

# 地形颜色
TERRAIN_COLORS = {
    0: (0, 0, 0),       # 空地 - 黑色
    1: (165, 42, 42),   # 砖墙 - 棕色
    2: (128, 128, 128), # 钢墙 - 灰色
    3: (0, 0, 255),     # 水域 - 蓝色
    4: (0, 128, 0)      # 草地 - 绿色
}

# 强化学习设置
RL_SETTINGS = {
    # 奖励设置
    "kill_reward": 100,       # 击杀敌方坦克奖励
    "hit_reward": 10,         # 击中敌方坦克奖励
    "hit_penalty": -5,        # 被击中惩罚
    "win_reward": 200,        # 胜利奖励
    "time_penalty": -0.1,     # 每步时间惩罚
    
    # 状态空间设置
    "observation_radius": 5,   # 观察半径（格子数）
    
    # 训练设置
    "max_steps": 1000,         # 每局最大步数
    "random_seed_range": (0, 10000)  # 随机种子范围
}

# 数据收集设置
DATA_COLLECTION = {
    "save_directory": "data/collected",
    "metrics": ["time", "kills", "hits", "damage_dealt", "damage_taken", "win"]
}