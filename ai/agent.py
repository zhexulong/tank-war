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
        map_input_size = int(np.prod(input_shape['map']))
        tank_input_size = int(np.prod(input_shape['tanks'])) # MODIFIED LINE
        bullet_input_size = int(np.prod(input_shape['bullets']))
        total_input_size = map_input_size + tank_input_size + bullet_input_size
        
        # 更深的全连接网络，增加层数和神经元
        self.fc_net = nn.Sequential(
            nn.Linear(total_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 384),
            nn.ReLU(),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x: Dict) -> torch.Tensor:
        # 展平所有输入
        map_flat = x['map'].view(x['map'].size(0), -1)
        tanks_flat = x['tanks'].view(x['tanks'].size(0), -1) # MODIFIED LINE
        bullets_flat = x['bullets'].view(x['bullets'].size(0), -1)
        
        # 合并所有特征
        combined = torch.cat([map_flat, tanks_flat, bullets_flat], dim=1)
        
        # 输出动作价值
        return self.fc_net(combined)

class ReplayBuffer:
    """支持多步学习的经验回放缓冲区"""
    
    def __init__(self, capacity: int, n_step: int = 3, gamma: float = 0.99):
        self.buffer = deque(maxlen=capacity)
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma
    
    def _calculate_n_step_info(self):
        """计算n步奖励和下一个状态"""
        reward = 0
        # 计算多步奖励
        for idx, (_, _, r, _, _) in enumerate(self.n_step_buffer):
            reward += r * (self.gamma ** idx)
        
        # 获取n步后的下一个状态和完成标志
        _, _, _, next_state, done = self.n_step_buffer[-1]
        
        # 获取初始状态和动作
        first_state, first_action, _, _, _ = self.n_step_buffer[0]
        
        return first_state, first_action, reward, next_state, done
    
    def push(self, state: Dict, action: int, reward: float, next_state: Dict, done: bool):
        """添加单步经验到n步缓冲区，并可能添加多步经验到主缓冲区"""
        # 将当前经验添加到n步缓冲区
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # 如果n步缓冲区未满，不添加到主缓冲区
        if len(self.n_step_buffer) < self.n_step:
            return
        
        # 计算n步奖励和状态
        first_state, first_action, n_reward, last_next_state, n_done = self._calculate_n_step_info()
        
        # 添加到主缓冲区
        self.buffer.append((first_state, first_action, n_reward, last_next_state, n_done))
        
        # 如果当前经验导致结束，清空n步缓冲区
        if done:
            self.n_step_buffer.clear()
    
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
        
        # 多步学习超参数
        self.n_step = 3  # n-step学习步数
        
        # 创建经验回放缓冲区(支持多步学习)
        self.replay_buffer = ReplayBuffer(10000, self.n_step, 0.99)
        
        # 超参数
        self.gamma = 0.99  # 折扣因子
        self.gamma_n = self.gamma ** self.n_step  # n步折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.999  # 探索率衰减
        self.target_update = 100  # 目标网络更新频率
        self.batch_size = 64  # 批量大小
        
        # 训练步数
        self.train_step = 0
    
    def select_action(self, state: Dict, training: bool = True) -> int:
        """选择动作"""
        # 打印原始状态以便调试
        print("DQNAgent raw state:", state)
        if training and np.random.rand() < self.epsilon:
            # 探索：随机选择动作
            return np.random.randint(self.action_dim)
        else:
            # 利用：选择价值最高的动作
            with torch.no_grad():
                # 转换状态为张量并打印输入tensor
                state_tensor = self._dict_to_tensor(state)
                print("DQNAgent input tensor:", state_tensor)
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
        
        # 计算目标Q值 (使用n步折扣因子)
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor).max(1, keepdim=True)[0]
            target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma_n * next_q_values
        
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
        """将状态字典转换为张量字典，确保输出符合 self.state_shape"""
        
        # Get expected shapes from self.state_shape
        expected_map_shape = self.state_shape['map']
        expected_tanks_shape = self.state_shape['tanks'] # (num_tanks, num_tank_features)
        expected_bullets_shape = self.state_shape['bullets'] # (num_bullets, num_bullet_features)

        # Define feature keys consistently (assuming these are correct from _batch_dict_to_tensor context)
        tank_feature_keys = ['x', 'y', 'angle', 'hp'] 
        bullet_feature_keys = ['x', 'y', 'angle'] # Consistent with _batch_dict_to_tensor summary

        # --- MAP PROCESSING ---
        map_np = np.zeros(expected_map_shape, dtype=np.float32)
        raw_map_data = state.get('map')
        if raw_map_data is not None:
            try:
                data_to_assign = np.asarray(raw_map_data, dtype=np.float32)
                # Determine slices to prevent out-of-bounds, ensuring data fits into map_np
                slices_source = tuple(slice(min(s_dim, e_dim)) for s_dim, e_dim in zip(data_to_assign.shape, expected_map_shape))
                slices_target = tuple(slice(min(s_dim, e_dim)) for s_dim, e_dim in zip(data_to_assign.shape, expected_map_shape))
                map_np[slices_target] = data_to_assign[slices_source]
            except Exception as e:
                # print(f"Warning: Could not process map data in _dict_to_tensor: {e}")
                pass # map_np remains zeros

        # --- TANKS PROCESSING ---
        tanks_np = np.zeros(expected_tanks_shape, dtype=np.float32)
        raw_tanks_data = state.get('tanks', [])
        if isinstance(raw_tanks_data, list):
            for i, tank_data_dict in enumerate(raw_tanks_data):
                if i < expected_tanks_shape[0]: # Ensure we don't exceed max_tanks
                    if isinstance(tank_data_dict, dict):
                        for j, key in enumerate(tank_feature_keys):
                            if j < expected_tanks_shape[1]: # Ensure we don't exceed tank_features
                                tanks_np[i, j] = float(tank_data_dict.get(key, 0.0))
                else:
                    break # Stop if we have more tanks in state than expected_tanks_shape allows

        # --- BULLETS PROCESSING ---
        bullets_np = np.zeros(expected_bullets_shape, dtype=np.float32)
        raw_bullets_data = state.get('bullets', [])
        if isinstance(raw_bullets_data, list):
            for i, bullet_data_dict in enumerate(raw_bullets_data):
                if i < expected_bullets_shape[0]: # Ensure we don't exceed max_bullets
                    if isinstance(bullet_data_dict, dict):
                        for j, key in enumerate(bullet_feature_keys):
                            if j < expected_bullets_shape[1]: # Ensure we don't exceed bullet_features
                                bullets_np[i, j] = float(bullet_data_dict.get(key, 0.0))
                else:
                    break # Stop if we have more bullets in state than expected_bullets_shape allows

        return {
            'map': torch.from_numpy(map_np).to(self.device).unsqueeze(0),
            'tanks': torch.from_numpy(tanks_np).to(self.device).unsqueeze(0),
            'bullets': torch.from_numpy(bullets_np).to(self.device).unsqueeze(0)
        }
    
    def _batch_dict_to_tensor(self, states: Tuple[Dict]) -> Dict:
        """将状态字典批量转换为张量字典"""
        
        expected_map_shape = self.state_shape['map']
        expected_tanks_shape = self.state_shape['tanks'] # (num_tanks, num_tank_features)
        expected_bullets_shape = self.state_shape['bullets'] # (num_bullets, num_bullet_features)

        tank_feature_keys = ['x', 'y', 'angle', 'hp'] # Should match expected_tanks_shape[1]
        bullet_feature_keys = ['x', 'y', 'angle'] # CORRECTED: Should match expected_bullets_shape[1]

        maps_list = []
        tanks_list = []
        bullets_list = []
        
        for s in states:
            # --- MAP PROCESSING ---
            raw_map_data = s.get('map')
            current_map_arr = np.zeros(expected_map_shape, dtype=np.float32)
            if raw_map_data is not None:
                try:
                    data_np = np.asarray(raw_map_data, dtype=np.float32)
                    # Handle cases where data_np might not be 2D or has incorrect shape
                    if data_np.ndim == expected_map_shape.ndim: # type: ignore
                        slice_dims = [min(data_np.shape[i], expected_map_shape[i]) for i in range(data_np.ndim)]
                        current_map_arr[tuple(slice(sd) for sd in slice_dims)] = data_np[tuple(slice(sd) for sd in slice_dims)]
                    elif data_np.size == np.prod(expected_map_shape): # type: ignore
                         current_map_arr = data_np.reshape(expected_map_shape)
                except Exception:
                    # Keep current_map_arr as zeros if conversion fails
                    pass
            maps_list.append(current_map_arr)

            # --- TANKS PROCESSING ---
            processed_tank_features_for_state = np.zeros(expected_tanks_shape, dtype=np.float32)
            raw_tanks_data = s.get('tanks', [])
            if isinstance(raw_tanks_data, list):
                for i in range(expected_tanks_shape[0]): # Iterate up to num_expected_tanks
                    if i < len(raw_tanks_data) and isinstance(raw_tanks_data[i], dict):
                        tank_dict = raw_tanks_data[i]
                        for j, key in enumerate(tank_feature_keys):
                            if j < expected_tanks_shape[1]: # Ensure we don't write out of bounds for features
                                processed_tank_features_for_state[i, j] = float(tank_dict.get(key, 0.0))
            tanks_list.append(processed_tank_features_for_state)

            # --- BULLETS PROCESSING ---
            processed_bullet_features_for_state = np.zeros(expected_bullets_shape, dtype=np.float32)
            raw_bullets_data = s.get('bullets', [])
            if isinstance(raw_bullets_data, list):
                for i in range(expected_bullets_shape[0]): # Iterate up to num_expected_bullets
                    if i < len(raw_bullets_data) and isinstance(raw_bullets_data[i], dict):
                        bullet_dict = raw_bullets_data[i]
                        for j, key in enumerate(bullet_feature_keys):
                             if j < expected_bullets_shape[1]: # Ensure we don't write out of bounds for features
                                processed_bullet_features_for_state[i, j] = float(bullet_dict.get(key, 0.0))
            bullets_list.append(processed_bullet_features_for_state)
        
        maps_np = np.stack(maps_list)
        tanks_np = np.stack(tanks_list)
        bullets_np = np.stack(bullets_list)

        return {
            'map': torch.from_numpy(maps_np).to(self.device),
            'tanks': torch.from_numpy(tanks_np).to(self.device),
            'bullets': torch.from_numpy(bullets_np).to(self.device)
        }
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'n_step': self.n_step,
            'gamma': self.gamma,
            'gamma_n': self.gamma_n
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']
        
        # 加载多步学习参数(兼容旧模型)
        if 'n_step' in checkpoint:
            self.n_step = checkpoint['n_step']
        if 'gamma' in checkpoint:
            self.gamma = checkpoint['gamma']
        if 'gamma_n' in checkpoint:
            self.gamma_n = checkpoint['gamma_n']
        else:
            self.gamma_n = self.gamma ** self.n_step

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