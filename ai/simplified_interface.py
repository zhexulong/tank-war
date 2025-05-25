import numpy as np
from typing import Dict
from ai.logic_agent import LogicAgent

class SimplifiedAIInterface:
    """简化版游戏的AI接口"""
    
    def __init__(self, game_manager, game_core):
        self.game_manager = game_manager
        self.game_core = game_core
        self.agents = {} 

    def load_agent(self, player_id: int, agent_path: str = None, agent_type: str = 'dqn'):
        """加载智能体"""
        if agent_type == 'logic':
            agent = LogicAgent()
        else:  # dqn
            from ai.agent import DQNAgent
            import torch
            import multiprocessing
            
            # 从游戏获取状态空间定义
            state_shape = self.game_manager.get_state_shape()
            action_dim = 6  # 不动/前进/后退/左转/右转/开火
            
            # 检测是否在子进程中运行
            if multiprocessing.current_process().name != 'MainProcess':
                # 工作进程中，强制使用CPU
                device = 'cpu'
            else:
                # 主进程中，如果可用则使用GPU
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            agent = DQNAgent(state_shape, action_dim, device=device)
            if agent_path:
                agent.load(agent_path)
        
        self.agents[player_id] = agent
    
    def get_observation(self, player_id: int) -> Dict:
        """获取观察"""
        tank = self.game_manager.get_player_tank(player_id)
        if not tank:
            return None
        
        # 直接使用游戏状态
        return self.game_manager.get_game_state_for_rl()

    def update_ai_controlled_tanks(self):
        """更新所有AI控制的坦克"""
        for player_id, agent in self.agents.items():
            tank = self.game_manager.get_player_tank(player_id)
            if not tank.is_player:  # AI控制的坦克
                observation = self.get_observation(player_id)
                if observation is None:
                    continue
                
                if isinstance(agent, LogicAgent):
                    action = agent.select_action(observation)
                else:
                    action = agent.select_action(observation)
                
                self._execute_action(tank, action)
    
    def _execute_action(self, tank, action):
        """执行AI的动作
        对于 LogicAgent:
        action: 0-不动, 1-前进, 2-左转, 3-右转, 4-开火
        
        对于 DQNAgent:  
        action: 0-不动, 1-前进, 2-后退, 3-左转, 4-右转, 5-开火
        """
        if isinstance(self.agents[tank.player_id], LogicAgent):
            # Logic agent 动作映射: 0-不动, 1-前进, 2-左转, 3-右转, 4-开火
            if action == 0:  # 不动
                pass
            elif action == 1:  # 前进
                new_pos = tank.position.copy()
                if tank.direction == 0:  # 上
                    new_pos[1] = max(0, new_pos[1] - 1)
                elif tank.direction == 1:  # 右
                    new_pos[0] = min(self.game_manager.MAP_SIZE - 1, new_pos[0] + 1)
                elif tank.direction == 2:  # 下
                    new_pos[1] = min(self.game_manager.MAP_SIZE - 1, new_pos[1] + 1)
                elif tank.direction == 3:  # 左
                    new_pos[0] = max(0, new_pos[0] - 1)
                
                # 检查碰撞
                if (new_pos[0], new_pos[1]) not in self.game_manager.current_map.obstacles:
                    tank.update_position(new_pos)
            elif action == 2:  # 左转
                tank.update_direction((tank.direction - 1) % 4)
            elif action == 3:  # 右转
                tank.update_direction((tank.direction + 1) % 4)
            elif action == 4:  # 开火
                self.game_manager.bullets.append([
                    tank.position[0],
                    tank.position[1], 
                    tank.direction,
                    tank.player_id
                ])
        else:
            # DQN agent 动作映射保持不变: 0-不动, 1-前进, 2-后退, 3-左转, 4-右转, 5-开火
            if action == 0:  # 不动
                pass
            elif action == 1:  # 前进
                new_pos = tank.position.copy()
                if tank.direction == 0:  # 上
                    new_pos[1] = max(0, new_pos[1] - 1)
                elif tank.direction == 1:  # 右
                    new_pos[0] = min(self.game_manager.MAP_SIZE - 1, new_pos[0] + 1)
                elif tank.direction == 2:  # 下
                    new_pos[1] = min(self.game_manager.MAP_SIZE - 1, new_pos[1] + 1)
                elif tank.direction == 3:  # 左
                    new_pos[0] = max(0, new_pos[0] - 1)
                
                # 检查碰撞
                if (new_pos[0], new_pos[1]) not in self.game_manager.current_map.obstacles:
                    tank.update_position(new_pos)
            elif action == 2:  # 后退
                new_pos = tank.position.copy()
                if tank.direction == 0:  # 上
                    new_pos[1] = min(self.game_manager.MAP_SIZE - 1, new_pos[1] + 1)
                elif tank.direction == 1:  # 右
                    new_pos[0] = max(0, new_pos[0] - 1)
                elif tank.direction == 2:  # 下
                    new_pos[1] = max(0, new_pos[1] - 1)
                elif tank.direction == 3:  # 左
                    new_pos[0] = min(self.game_manager.MAP_SIZE - 1, new_pos[0] + 1)
                
                # 检查碰撞
                if (new_pos[0], new_pos[1]) not in self.game_manager.current_map.obstacles:
                    tank.update_position(new_pos)
            elif action == 3:  # 左转
                tank.update_direction((tank.direction - 1) % 4)
            elif action == 4:  # 右转
                tank.update_direction((tank.direction + 1) % 4)
            elif action == 5:  # 开火
                self.game_manager.bullets.append([
                    tank.position[0],
                    tank.position[1], 
                    tank.direction,
                    tank.player_id
                ])
