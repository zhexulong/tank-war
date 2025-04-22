import gym
import numpy as np
from typing import Dict, Tuple, List, Any, Optional
import pygame

from game.game_manager import GameManager
from game.config import SCREEN_WIDTH, SCREEN_HEIGHT, RL_SETTINGS, TERRAIN_TYPES

class TankBattleEnv(gym.Env):
    """坦克大战强化学习环境"""
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, render_mode: str = None):
        super().__init__()
        
        # 渲染模式
        self.render_mode = render_mode
        
        # 初始化pygame（如果尚未初始化）
        if not pygame.get_init():
            pygame.init()
        
        # 创建屏幕（如果需要渲染）
        if self.render_mode == 'human':
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("坦克大战 - RL训练")
        else:
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        
        # 创建游戏管理器
        self.game_manager = None
        
        # 定义动作空间
        # 动作空间包括：前进、后退、左转、右转、炮塔左转、炮塔右转、开火、不动
        # 0: 不动, 1: 前进, 2: 后退, 3: 左转, 4: 右转, 5: 炮塔左转, 6: 炮塔右转, 7: 开火
        self.action_space = gym.spaces.Discrete(8)
        
        # 定义观察空间
        # 观察半径内的地图状态 + 坦克状态 + 子弹状态
        obs_radius = RL_SETTINGS["observation_radius"]
        map_shape = (obs_radius * 2 + 1, obs_radius * 2 + 1)  # 观察区域的地图
        
        # 观察空间包括：
        # 1. 地图状态 (obs_radius*2+1 x obs_radius*2+1 x 5) - 5种地形类型的one-hot编码
        # 2. 坦克状态 (10) - 自身位置(2)、方向(1)、炮塔方向(1)、生命值(1)、弹药(1)、装填进度(1)、敌方位置(2)、敌方方向(1)
        # 3. 子弹状态 (最多10颗子弹，每颗5个特征) - 位置(2)、方向(1)、伤害(1)、所有者(1)
        self.observation_space = gym.spaces.Dict({
            'map': gym.spaces.Box(low=0, high=1, shape=(*map_shape, 5), dtype=np.uint8),
            'tanks': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
            'bullets': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10, 5), dtype=np.float32)
        })
        
        # 当前步数
        self.current_step = 0
        
        # 最大步数
        self.max_steps = RL_SETTINGS["max_steps"]
        
        # 奖励设置
        self.kill_reward = RL_SETTINGS["kill_reward"]
        self.hit_reward = RL_SETTINGS["hit_reward"]
        self.hit_penalty = RL_SETTINGS["hit_penalty"]
        self.win_reward = RL_SETTINGS["win_reward"]
        self.time_penalty = RL_SETTINGS["time_penalty"]
        
        # 上一步的游戏状态（用于计算奖励）
        self.last_game_state = None
    
    def reset(self, seed: Optional[int] = None) -> Dict:
        """重置环境"""
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
            random_seed = seed
        else:
            random_seed = np.random.randint(*RL_SETTINGS["random_seed_range"])
        
        # 重新创建游戏管理器
        self.game_manager = GameManager(self.screen)
        
        # 重置步数
        self.current_step = 0
        
        # 获取初始观察
        observation = self._get_observation()
        
        # 保存当前游戏状态
        self.last_game_state = self.game_manager.game_state.copy()
        
        return observation
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """执行一步动作"""
        # 增加步数
        self.current_step += 1
        
        # 执行动作
        self._execute_action(action)
        
        # 更新游戏状态
        self.game_manager.update()
        
        # 获取观察
        observation = self._get_observation()
        
        # 计算奖励
        reward = self._calculate_reward()
        
        # 检查是否结束
        done = self.game_manager.game_over or self.current_step >= self.max_steps
        
        # 保存当前游戏状态
        self.last_game_state = self.game_manager.game_state.copy()
        
        # 返回信息
        info = {
            'current_step': self.current_step,
            'max_steps': self.max_steps,
            'game_over': self.game_manager.game_over,
            'winner': self.game_manager.winner
        }
        
        return observation, reward, done, info
    
    def render(self, mode: str = 'human'):
        """渲染环境"""
        if self.game_manager is None:
            return
        
        # 渲染游戏
        self.game_manager.render()
        
        if mode == 'human':
            pygame.display.flip()
        elif mode == 'rgb_array':
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
    
    def close(self):
        """关闭环境"""
        if hasattr(self, 'screen') and self.render_mode == 'human':
            pygame.quit()
    
    def _execute_action(self, action: int):
        """执行动作"""
        # 获取玩家1的坦克（AI控制）
        tank = self.game_manager.get_player_tank(1)
        if tank is None:
            return
        
        # 重置所有控制状态
        tank.moving_forward = False
        tank.moving_backward = False
        tank.rotating_left = False
        tank.rotating_right = False
        tank.rotating_turret_left = False
        tank.rotating_turret_right = False
        tank.firing = False
        
        # 根据动作设置控制状态
        if action == 0:  # 不动
            pass
        elif action == 1:  # 前进
            tank.moving_forward = True
        elif action == 2:  # 后退
            tank.moving_backward = True
        elif action == 3:  # 左转
            tank.rotating_left = True
        elif action == 4:  # 右转
            tank.rotating_right = True
        elif action == 5:  # 炮塔左转
            tank.rotating_turret_left = True
        elif action == 6:  # 炮塔右转
            tank.rotating_turret_right = True
        elif action == 7:  # 开火
            tank.firing = True
    
    def _get_observation(self) -> Dict:
        """获取观察"""
        if self.game_manager is None:
            return {}
        
        # 获取玩家1的坦克（AI控制）
        tank = self.game_manager.get_player_tank(1)
        if tank is None:
            # 如果坦克不存在，返回零观察
            return {
                'map': np.zeros(self.observation_space['map'].shape, dtype=np.uint8),
                'tanks': np.zeros(self.observation_space['tanks'].shape, dtype=np.float32),
                'bullets': np.zeros(self.observation_space['bullets'].shape, dtype=np.float32)
            }
        
        # 获取地图数据
        map_data = self._get_map_observation(tank)
        
        # 获取坦克数据
        tank_data = self._get_tank_observation(tank)
        
        # 获取子弹数据
        bullet_data = self._get_bullet_observation()
        
        return {
            'map': map_data,
            'tanks': tank_data,
            'bullets': bullet_data
        }
    
    def _get_map_observation(self, tank) -> np.ndarray:
        """获取地图观察"""
        # 获取观察半径
        obs_radius = RL_SETTINGS["observation_radius"]
        
        # 创建地图观察数组
        map_obs = np.zeros((obs_radius * 2 + 1, obs_radius * 2 + 1, 5), dtype=np.uint8)
        
        # 获取坦克位置（格子坐标）
        tank_tile_x = int(tank.position[0] / self.game_manager.current_map.tile_size)
        tank_tile_y = int(tank.position[1] / self.game_manager.current_map.tile_size)
        
        # 填充地图观察
        for dx in range(-obs_radius, obs_radius + 1):
            for dy in range(-obs_radius, obs_radius + 1):
                x, y = tank_tile_x + dx, tank_tile_y + dy
                obs_x, obs_y = dx + obs_radius, dy + obs_radius
                
                # 检查是否在地图范围内
                if 0 <= x < self.game_manager.current_map.width and 0 <= y < self.game_manager.current_map.height:
                    tile_type = self.game_manager.current_map.get_tile(x, y)
                    
                    # 设置one-hot编码
                    if tile_type == TERRAIN_TYPES["empty"]:
                        map_obs[obs_y, obs_x, 0] = 1
                    elif tile_type == TERRAIN_TYPES["brick"]:
                        map_obs[obs_y, obs_x, 1] = 1
                    elif tile_type == TERRAIN_TYPES["steel"]:
                        map_obs[obs_y, obs_x, 2] = 1
                    elif tile_type == TERRAIN_TYPES["water"]:
                        map_obs[obs_y, obs_x, 3] = 1
                    elif tile_type == TERRAIN_TYPES["grass"]:
                        map_obs[obs_y, obs_x, 4] = 1
                else:
                    # 地图外的区域标记为钢墙
                    map_obs[obs_y, obs_x, 2] = 1
        
        return map_obs
    
    def _get_tank_observation(self, tank) -> np.ndarray:
        """获取坦克观察"""
        # 创建坦克观察数组
        tank_obs = np.zeros(10, dtype=np.float32)
        
        # 填充自身坦克数据
        tank_obs[0] = tank.position[0] / SCREEN_WIDTH  # x位置（归一化）
        tank_obs[1] = tank.position[1] / SCREEN_HEIGHT  # y位置（归一化）
        tank_obs[2] = tank.direction / 360.0  # 方向（归一化）
        tank_obs[3] = tank.turret_direction / 360.0  # 炮塔方向（归一化）
        tank_obs[4] = tank.health / tank.max_health  # 生命值（归一化）
        tank_obs[5] = tank.ammo / 10.0  # 弹药（假设最大10，归一化）
        tank_obs[6] = tank.reload_progress  # 装填进度
        
        # 获取敌方坦克
        enemy_tank = None
        for t in self.game_manager.tanks:
            if t.player_id != tank.player_id:
                enemy_tank = t
                break
        
        # 填充敌方坦克数据
        if enemy_tank is not None:
            tank_obs[7] = enemy_tank.position[0] / SCREEN_WIDTH  # 敌方x位置（归一化）
            tank_obs[8] = enemy_tank.position[1] / SCREEN_HEIGHT  # 敌方y位置（归一化）
            tank_obs[9] = enemy_tank.direction / 360.0  # 敌方方向（归一化）
        else:
            # 如果没有敌方坦克，设置为-1
            tank_obs[7:10] = -1
        
        return tank_obs
    
    def _get_bullet_observation(self) -> np.ndarray:
        """获取子弹观察"""
        # 创建子弹观察数组（最多10颗子弹）
        bullet_obs = np.zeros((10, 5), dtype=np.float32)
        
        # 填充子弹数据
        for i, bullet in enumerate(self.game_manager.bullets[:10]):
            bullet_obs[i, 0] = bullet.position[0] / SCREEN_WIDTH  # x位置（归一化）
            bullet_obs[i, 1] = bullet.position[1] / SCREEN_HEIGHT  # y位置（归一化）
            bullet_obs[i, 2] = bullet.direction / 360.0  # 方向（归一化）
            bullet_obs[i, 3] = bullet.damage / 50.0  # 伤害（假设最大50，归一化）
            bullet_obs[i, 4] = bullet.owner.player_id  # 所有者ID
        
        return bullet_obs
    
    def _calculate_reward(self) -> float:
        """计算奖励"""
        if self.game_manager is None or self.last_game_state is None:
            return 0.0
        
        reward = 0.0
        
        # 获取当前游戏状态
        current_state = self.game_manager.game_state
        
        # 击杀奖励
        kills_diff = current_state["player1_kills"] - self.last_game_state["player1_kills"]
        reward += kills_diff * self.kill_reward
        
        # 命中奖励
        hits_diff = current_state["player1_hits"] - self.last_game_state["player1_hits"]
        reward += hits_diff * self.hit_reward
        
        # 被命中惩罚
        damage_taken_diff = current_state["player1_damage_taken"] - self.last_game_state["player1_damage_taken"]
        reward += damage_taken_diff * self.hit_penalty
        
        # 胜利奖励
        if self.game_manager.game_over and self.game_manager.winner == 1:
            reward += self.win_reward
        
        # 时间惩罚
        reward += self.time_penalty
        
        return reward