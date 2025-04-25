import pygame
import numpy as np
from typing import Dict, List, Tuple, Optional

from game.entities.tank import Tank
from game.game_manager import GameManager
from game.core import GameCore
from ai.agent import DQNAgent

class AIInterface:
    """AI接口类，连接强化学习环境与游戏"""
    
    def __init__(self, game_manager: GameManager, game_core: GameCore):
        self.game_manager = game_manager
        self.game_core = game_core
        self.agents = {}  # 存储AI智能体 {player_id: agent}
        self.observation_cache = {}  # 缓存观察结果
    
    def load_agent(self, player_id: int, agent_path: str = None, agent_type: str = 'dqn'):
        """加载智能体
        
        Args:
            player_id: 玩家ID
            agent_path: 预训练模型路径（仅DQN需要）
            agent_type: 智能体类型，'dqn'或'logic'
        """
        if agent_type == 'logic':
            from ai.logic_agent import LogicAgent
            agent = LogicAgent()
        else:  # dqn
            # 获取状态和动作空间
            from ai.environment import TankBattleEnv
            env = TankBattleEnv()
            state_shape = {
                'map': env.observation_space['map'].shape,
                'tanks': env.observation_space['tanks'].shape,
                'bullets': env.observation_space['bullets'].shape
            }
            action_dim = env.action_space.n
            
            # 创建智能体
            agent = DQNAgent(state_shape, action_dim)
            
            # 加载预训练模型
            if agent_path:
                agent.load(agent_path)
        
        # 存储智能体
        self.agents[player_id] = agent
    
    def get_observation(self, player_id: int) -> Dict:
        """获取指定玩家的观察"""
        # 获取玩家坦克
        tank = self.game_manager.get_player_tank(player_id)
        if not tank:
            return None
        
        # 获取游戏状态
        game_state = self.game_manager.get_game_state_for_rl()
        
        # 构建观察
        observation = self._build_observation(game_state, tank)
        
        # 缓存观察
        self.observation_cache[player_id] = observation
        
        return observation
    
    def _build_observation(self, game_state: Dict, tank: Tank) -> Dict:
        """构建观察"""
        from game.config import RL_SETTINGS
        
        # 观察半径
        obs_radius = RL_SETTINGS["observation_radius"]
        
        # 构建地图观察
        map_data = np.zeros((obs_radius * 2 + 1, obs_radius * 2 + 1, 5), dtype=np.uint8)
        
        # 获取坦克所在的地图格子
        tank_tile_x = int(tank.position[0] / self.game_manager.current_map.tile_size)
        tank_tile_y = int(tank.position[1] / self.game_manager.current_map.tile_size)
        
        # 填充地图数据
        for y in range(-obs_radius, obs_radius + 1):
            for x in range(-obs_radius, obs_radius + 1):
                map_x = tank_tile_x + x
                map_y = tank_tile_y + y
                
                # 检查是否在地图范围内
                if 0 <= map_x < self.game_manager.current_map.width and 0 <= map_y < self.game_manager.current_map.height:
                    tile_type = self.game_manager.current_map.get_tile(map_x, map_y)
                    map_data[y + obs_radius, x + obs_radius, tile_type] = 1
                else:
                    # 地图边界外设为钢墙
                    map_data[y + obs_radius, x + obs_radius, 2] = 1
        
        # 构建坦克观察
        tanks_data = np.zeros(10, dtype=np.float32)
        
        # 自身信息
        tanks_data[0] = tank.position[0] / self.game_manager.current_map.width / self.game_manager.current_map.tile_size  # x位置归一化
        tanks_data[1] = tank.position[1] / self.game_manager.current_map.height / self.game_manager.current_map.tile_size  # y位置归一化
        tanks_data[2] = tank.direction / 360.0  # 方向归一化
        tanks_data[3] = tank.turret_direction / 360.0  # 炮塔方向归一化
        tanks_data[4] = tank.health / tank.max_health  # 生命值归一化
        tanks_data[5] = tank.ammo / 10.0  # 弹药归一化
        tanks_data[6] = tank.reload_progress  # 装填进度
        
        # 敌方信息
        enemy_tanks = [t for t in self.game_manager.tanks if t.player_id != tank.player_id]
        if enemy_tanks:
            enemy = enemy_tanks[0]  # 取第一个敌人
            tanks_data[7] = enemy.position[0] / self.game_manager.current_map.width / self.game_manager.current_map.tile_size  # 敌人x位置
            tanks_data[8] = enemy.position[1] / self.game_manager.current_map.height / self.game_manager.current_map.tile_size  # 敌人y位置
            tanks_data[9] = enemy.direction / 360.0  # 敌人方向
        
        # 构建子弹观察
        bullets_data = np.zeros((10, 5), dtype=np.float32)
        
        # 填充子弹数据
        for i, bullet in enumerate(self.game_manager.bullets[:10]):  # 最多观察10颗子弹
            bullets_data[i, 0] = bullet.position[0] / self.game_manager.current_map.width / self.game_manager.current_map.tile_size  # x位置
            bullets_data[i, 1] = bullet.position[1] / self.game_manager.current_map.height / self.game_manager.current_map.tile_size  # y位置
            bullets_data[i, 2] = bullet.direction / 360.0  # 方向
            bullets_data[i, 3] = bullet.damage / 50.0  # 伤害归一化
            bullets_data[i, 4] = 1.0 if bullet.owner.player_id == tank.player_id else 0.0  # 所有者
        
        return {
            'map': map_data,
            'tanks': tanks_data,
            'bullets': bullets_data
        }
    
    def take_action(self, player_id: int, action: int):
        """执行动作"""
        # 获取玩家坦克
        tank = self.game_manager.get_player_tank(player_id)
        if not tank:
            return
        
        # 重置控制状态
        tank.moving_forward = False
        tank.moving_backward = False
        tank.rotating_left = False
        tank.rotating_right = False
        tank.rotating_turret_left = False
        tank.rotating_turret_right = False
        tank.firing = False
        
        # 根据动作设置控制状态
        # 0: 不动, 1: 前进, 2: 后退, 3: 左转, 4: 右转, 5: 炮塔左转, 6: 炮塔右转, 7: 开火
        if action == 1:
            tank.moving_forward = True
        elif action == 2:
            tank.moving_backward = True
        elif action == 3:
            tank.rotating_left = True
        elif action == 4:
            tank.rotating_right = True
        elif action == 5:
            tank.rotating_turret_left = True
        elif action == 6:
            tank.rotating_turret_right = True
        elif action == 7:
            tank.firing = True
    
    def update_ai_controlled_tanks(self):
        """更新AI控制的坦克"""
        for player_id, agent in self.agents.items():
            # 获取观察
            observation = self.get_observation(player_id)
            if observation is None:
                print(f"警告: 坦克 {player_id} 的观察数据为空")
                continue
            
            print(f"\n坦克 {player_id} 的观察数据:")
            print(f"- 坦克数据: {observation['tanks']}")
            print(f"- 地图数据形状: {observation['map'].shape}")
            print(f"- 子弹数量: {len([b for b in observation['bullets'] if any(b)])}")
            
            # 选择动作
            action = agent.select_action(observation, epsilon=0.05)
            print(f"- 选择的动作: {action}")
            
            # 执行动作
            self.take_action(player_id, action)
            
            # 获取坦克当前状态
            tank = self.game_manager.get_player_tank(player_id)
            if tank:
                print(f"- 坦克状态: 前进={tank.moving_forward}, 后退={tank.moving_backward}, " 
                      f"左转={tank.rotating_left}, 右转={tank.rotating_right}, "
                      f"炮塔左转={tank.rotating_turret_left}, 炮塔右转={tank.rotating_turret_right}, "
                      f"开火={tank.firing}")
    
    def run_ai_vs_ai_game(self, agent1_path: str, agent2_path: str, render: bool = True) -> int:
        """运行AI对战游戏"""
        # 加载智能体
        self.load_agent(1, agent1_path)
        self.load_agent(2, agent2_path)
        
        # 设置所有坦克为AI控制
        for tank in self.game_manager.tanks:
            tank.is_player = False
        
        # 游戏循环
        clock = pygame.time.Clock()
        running = True
        
        while running and not self.game_manager.game_over:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # 更新AI控制的坦克
            self.update_ai_controlled_tanks()
            
            # 更新游戏状态
            self.game_manager.update()
            
            # 渲染游戏
            if render:
                self.game_manager.screen.fill((0, 0, 0))
                self.game_manager.render()
                pygame.display.flip()
            
            # 控制帧率
            clock.tick(60)
        
        return self.game_manager.winner