import numpy as np
from typing import Dict, List, Tuple
from game_simplified.game import SimplifiedGame
from ai.base_agent import BaseAgent

class SimplifiedGameEnv:
    """简化版游戏环境"""
    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.game = None
        
        # 定义观察空间
        self.observation_space = {
            'map': (16, 16),  # 地图大小，与 SimplifiedGame.MAP_SIZE 保持一致
            'tanks': (2, 4),  # 2个坦克，每个坦克4个属性（x, y, angle, health）
            'bullets': (10, 3)  # 最多10颗子弹，每颗子弹3个属性（x, y, angle）
        }
        
        # 定义动作空间
        self.action_space = type('', (), {'n': 6})()  # 5个动作：0-不动，1-前进，2-后退，3-左转，4-右转
        
        # 确保两个智能体都被初始化
        self.ai_opponent = True
        self.ai_type = 'logic'
        self.second_ai_type = 'dqn'
        self.reset()
        # 定义状态格式化方法
    def _format_state(self, raw_state: Dict) -> Dict:
        # 将原始字典状态转换为 DQNAgent 可识别的键值格式
        # 获取最大容量和特征维度
        max_map = self.observation_space['map']  # (h, w)
        max_tanks, tank_feats = self.observation_space['tanks']
        max_bullets, bullet_feats = self.observation_space['bullets']
        # 地图
        formatted = {
            'map': raw_state.get('map', [])
        }
        # 坦克列表，pad/truncate到固定数量
        formatted['tanks'] = []
        raw_tanks = raw_state.get('tanks', [])
        for i in range(max_tanks):
            if i < len(raw_tanks):
                t = raw_tanks[i]
                formatted['tanks'].append({
                    'x': t.get('position', [0,0])[0],
                    'y': t.get('position', [0,0])[1],
                    'angle': t.get('direction', 0),
                    'player_id': t.get('player_id', 1.0)
                })
            else:
                formatted['tanks'].append({'x':0,'y':0,'angle':0,'player_id':0})
        # 子弹列表，pad/truncate到固定数量
        formatted['bullets'] = []
        raw_bullets = raw_state.get('bullets', [])
        for i in range(max_bullets):
            if i < len(raw_bullets):
                b = raw_bullets[i]
                # b: [x, y, angle, owner]
                formatted['bullets'].append({
                    'x': b[0],
                    'y': b[1],
                    'angle': b[2]
                })
            else:
                formatted['bullets'].append({'x':0,'y':0,'angle':0})
        formatted['obstacles'] = raw_state.get('obstacles', [])
        return formatted

    def reset(self):
        """重置环境"""
        # 创建新游戏实例
        self.game = SimplifiedGame(ai_opponent=True, ai_type=self.ai_type, second_ai_type=self.second_ai_type, render_mode=self.render_mode)
        
        # 确保两个智能体都被正确初始化
        self.game.ai_interface.load_agent(1, None, self.second_ai_type)  # DQN agent
        self.game.ai_interface.load_agent(2, None, self.ai_type)  # Logic agent
        
        # 获取并格式化初始状态
        raw_state = self.game.ai_interface.get_observation(1)
        return self._format_state(raw_state)
    
    def step(self, actions):
        """执行动作并返回下一个状态
        
        Args:
            actions: [rl_action, logic_action] 两个智能体的动作
            
        Returns:
            next_state: 下一个状态
            rewards: 奖励列表 [rl_reward, logic_reward]
            done: 游戏是否结束
            info: 附加信息
        """
        # 获取当前状态
        state = self.game.ai_interface.get_observation(1)
        
        # 执行RL智能体的动作
        tank1 = self.game.get_player_tank(1)
        self.game.ai_interface._execute_action(tank1, actions[0])
          # 执行Logic智能体的动作
        tank2 = self.game.get_player_tank(2)
        self.game.ai_interface._execute_action(tank2, actions[1])
        
        # 更新游戏状态
        # self.game.update_ai()
        self.game.update_bullets()
        
        # 获取新状态
        next_state = self.game.ai_interface.get_observation(1)
        
        # 计算奖励
        rewards = self._compute_rewards(state, next_state)
        
        # 获取游戏结束状态和胜者信息
        done = next_state['game_over']
        info = {'winner': next_state['winner']}
        
        # 渲染
        if self.render_mode == 'human':
            self.game.draw()
        
        # 格式化下一个状态
        formatted_next = self._format_state(next_state)
        return formatted_next, rewards, done, info
    
    def _compute_rewards(self, state: Dict, next_state: Dict) -> List[float]:
        """计算奖励
        
        Args:
            state: 当前状态
            next_state: 下一个状态
            
        Returns:
            rewards: [rl_reward, logic_reward] 奖励列表
        """
        rl_reward = 0
        logic_reward = 0
        
        # 游戏结束奖励
        if next_state['game_over']:
            if next_state['winner'] == 1:  # RL智能体获胜
                rl_reward += 50
                logic_reward -= 50
            elif next_state['winner'] == 2:  # Logic智能体获胜
                rl_reward -= 50
                logic_reward += 50
        
        # 存活奖励
        rl_reward += 0.1
        logic_reward += 0.1
        
        return [rl_reward, logic_reward]
    
    def render(self):
        """渲染游戏画面"""
        if self.render_mode == 'human':
            self.game.draw()
    
    def close(self):
        """关闭环境"""
        if self.game:
            self.game.close()