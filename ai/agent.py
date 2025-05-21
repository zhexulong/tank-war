import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple
from collections import deque
import random

class DQNNetwork(nn.Module):
    """轻量级深度Q网络，用于估计动作价值"""
    
    def __init__(self, input_shape: Dict, output_dim: int):
        super(DQNNetwork, self).__init__()
        
        # 计算展平后的输入维度
        map_input_size = input_shape['map'][0] * input_shape['map'][1] * input_shape['map'][2]
        tank_input_size = input_shape['tanks'][0]
        bullet_input_size = input_shape['bullets'][0] * input_shape['bullets'][1]
        total_input_size = map_input_size + tank_input_size + bullet_input_size
        
        # 简单的全连接网络
        self.fc_net = nn.Sequential(
            nn.Linear(total_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x: Dict) -> torch.Tensor:
        # 展平所有输入
        map_flat = x['map'].view(x['map'].size(0), -1)
        tanks_flat = x['tanks']
        bullets_flat = x['bullets'].view(x['bullets'].size(0), -1)
        
        # 合并所有特征
        combined = torch.cat([map_flat, tanks_flat, bullets_flat], dim=1)
        
        # 输出动作价值
        return self.fc_net(combined)

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: Dict, action: int, reward: float, next_state: Dict, done: bool):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List:
        """随机采样经验"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)

from ai.base_agent import BaseAgent

class DQNAgent(BaseAgent):
    """DQN智能体"""
    
    def __init__(self, state_shape: Dict, action_dim: int, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.device = device
        
        # 创建Q网络和目标网络
        self.q_network = DQNNetwork(state_shape, action_dim).to(device)
        self.target_network = DQNNetwork(state_shape, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # 创建优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0001)
        
        # 创建经验回放缓冲区
        self.replay_buffer = ReplayBuffer(10000)
        
        # 超参数
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.1  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        self.target_update = 10  # 目标网络更新频率
        self.batch_size = 64  # 批量大小
        
        # 训练步数
        self.train_step = 0
    
    def select_action(self, state: Dict, training: bool = True) -> int:
        """选择动作"""
        if training and np.random.rand() < self.epsilon:
            # 探索：随机选择动作
            return np.random.randint(self.action_dim)
        else:
            # 利用：选择价值最高的动作
            with torch.no_grad():
                # 转换状态为张量
                state_tensor = self._dict_to_tensor(state)
                
                # 获取动作价值
                q_values = self.q_network(state_tensor)
                
                # 选择价值最高的动作
                return q_values.argmax().item()
    
    def train(self):
        """训练智能体"""
        # 检查经验回放缓冲区是否有足够的样本
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 采样经验
        batch = self.replay_buffer.sample(self.batch_size)
        
        # 解包经验
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为张量
        states_tensor = self._batch_dict_to_tensor(states)
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_tensor = self._batch_dict_to_tensor(next_states)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        # 计算当前Q值
        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor).max(1, keepdim=True)[0]
            target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values
        
        # 计算损失
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.train_step += 1
        if self.train_step % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def _dict_to_tensor(self, state: Dict) -> Dict:
        """将状态字典转换为张量字典"""
        return {
            'map': torch.tensor(state['map'], dtype=torch.float32, device=self.device).unsqueeze(0),
            'tanks': torch.tensor(state['tanks'], dtype=torch.float32, device=self.device).unsqueeze(0),
            'bullets': torch.tensor(state['bullets'], dtype=torch.float32, device=self.device).unsqueeze(0)
        }
    
    def _batch_dict_to_tensor(self, states: Tuple[Dict]) -> Dict:
        """将状态字典批量转换为张量字典"""
        return {
            'map': torch.tensor(np.array([s['map'] for s in states]), dtype=torch.float32, device=self.device),
            'tanks': torch.tensor(np.array([s['tanks'] for s in states]), dtype=torch.float32, device=self.device),
            'bullets': torch.tensor(np.array([s['bullets'] for s in states]), dtype=torch.float32, device=self.device)
        }
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']

class MultiAgentDQN:
    """多智能体DQN"""
    
    def __init__(self, state_shape: Dict, action_dim: int, num_agents: int = 2, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.agents = [DQNAgent(state_shape, action_dim, device) for _ in range(num_agents)]
    
    def select_actions(self, states: List[Dict], training: bool = True) -> List[int]:
        """为所有智能体选择动作"""
        return [agent.select_action(state, training) for agent, state in zip(self.agents, states)]
    
    def train(self):
        """训练所有智能体"""
        return [agent.train() for agent in self.agents]
    
    def save(self, paths: List[str]):
        """保存所有智能体模型"""
        for agent, path in zip(self.agents, paths):
            agent.save(path)
    
    def load(self, paths: List[str]):
        """加载所有智能体模型"""
        for agent, path in zip(self.agents, paths):
            agent.load(path)