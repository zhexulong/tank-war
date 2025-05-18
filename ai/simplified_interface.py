import numpy as np
from typing import Dict
from ai.logic_agent import LogicAgent

class SimplifiedAIInterface:
    """简化版游戏的AI接口"""
    
    def __init__(self, game_manager, game_core):
        self.game_manager = game_manager  # 确保传入 game_manager
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
            # 在创建 LogicAgent 时传入 game_manager
            agent = LogicAgent(self.game_manager)
        else:  # dqn
            from ai.agent import DQNAgent
            
            # 定义简化版的状态和动作空间
            state_shape = {
                'map': (7, 7, 3),  # 简化的地图观察范围
                'tanks': 6,  # 简化的坦克状态信息
            }
            action_dim = 6  # 上下左右移动、旋转和开火
            
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

    def _build_observation(self, game_state: Dict, tank) -> Dict:
        """构建简化版的观察空间"""
        # 观察范围 (7x7)
        obs_radius = 3
        map_size = self.game_manager.MAP_SIZE
        
        # 构建地图观察 (空地=0, 障碍=1, 边界=2)
        map_data = np.zeros((obs_radius * 2 + 1, obs_radius * 2 + 1, 3), dtype=np.uint8)
        
        tank_x, tank_y = tank.position
        
        # 填充地图数据
        for y in range(-obs_radius, obs_radius + 1):
            for x in range(-obs_radius, obs_radius + 1):
                map_x = tank_x + x
                map_y = tank_y + y
                
                # 检查是否在地图范围内
                if 0 <= map_x < map_size and 0 <= map_y < map_size:
                    # 检查障碍物
                    if (map_x, map_y) in self.game_manager.current_map.obstacles:
                        map_data[y + obs_radius, x + obs_radius, 1] = 1
                    else:
                        map_data[y + obs_radius, x + obs_radius, 0] = 1
                else:
                    # 地图边界
                    map_data[y + obs_radius, x + obs_radius, 2] = 1
        
        # 构建坦克数据
        tanks_data = np.zeros(6, dtype=np.float32)
        
        # 当前坦克位置和方向
        tanks_data[0] = tank_x   # x位置归一化
        tanks_data[1] = tank_y   # y位置归一化
        tanks_data[2] = tank.direction / 4.0  # 方向归一化 (4个方向)
        
        # 获取对手坦克
        opponent_id = 1 if tank.player_id == 2 else 2
        opponent = None
        for t in game_state['tanks']:
            if t['player_id'] == opponent_id:
                opponent = t
                break
        
        if opponent:
            # 对手坦克位置和方向
            tanks_data[3] = opponent['position'][0] 
            tanks_data[4] = opponent['position'][1] 
            tanks_data[5] = opponent['direction'] / 4.0
        
        #添加子弹信息
        bullets_data=game_state['bullets']
        # print(f'bullets_data{bullets_data}')
        return {
            'map': map_data,
            'tanks': tanks_data,
            'game_over': game_state['game_over'],
            'winner': game_state['winner'],
            'bullets':bullets_data
        }
    
    def update_ai_controlled_tanks(self):
        """更新所有AI控制的坦克"""
        for player_id, agent in self.agents.items():
            tank = self.game_manager.get_player_tank(player_id)
            if not tank.is_player:  # 如果是AI控制的坦克
                # 获取观察
                observation = self.get_observation(player_id)
                if observation is None:
                    continue
                
                # 获取动作
                if isinstance(agent, LogicAgent):
                    action = agent.select_action(observation)
                else:
                    action = agent.get_action(observation)
                
                # 执行动作
                self._execute_action(tank, action)
    
    def _execute_action(self, tank, action):
        """执行AI的动作"""
        if action == 0:  # 不动
            pass
        elif action == 1:  # 向前移动
            # 检查前方是否有障碍物
            new_pos = tank.position.copy()
            if tank.direction == 0:  # 上
                new_pos[1] = max(0, new_pos[1] - 1)
            elif tank.direction == 1:  # 右
                new_pos[0] = min(self.game_manager.MAP_SIZE - 1, new_pos[0] + 1)
            elif tank.direction == 2:  # 下
                new_pos[1] = min(self.game_manager.MAP_SIZE - 1, new_pos[1] + 1)
            elif tank.direction == 3:  # 左
                new_pos[0] = max(0, new_pos[0] - 1)
            tank.update_position(new_pos)
        elif action == 2:  # 旋转（逆时针）
            tank.update_direction((tank.direction - 1) % 4)
        elif action == 3:  # 旋转（顺时针）
            tank.update_direction((tank.direction + 1) % 4)
        elif action == 4:  # 开火
            self._fire(tank)  # 开火
        
    def _fire(self, tank):
        """执行开火动作"""
        print(f"坦克 {tank.player_id} 在位置 {tank.position} 发射子弹！")
        self.game_manager.bullets.append([
            tank.position[0],
            tank.position[1],
            tank.direction,
            tank.player_id
        ])
        