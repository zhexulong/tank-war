import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple
from collections import deque
import random

from ai.base_agent import BaseAgent


class PPONetwork(nn.Module):
    """PPO策略-价值网络"""
    
    def __init__(self, input_shape: Dict, action_dim: int):
        super(PPONetwork, self).__init__()
        
        # 计算输入维度
        map_size = np.prod(input_shape['map'])
        tanks_size = np.prod(input_shape['tanks'])
        bullets_size = np.prod(input_shape['bullets'])
        total_input_size = map_size + tanks_size + bullets_size
        
        # 共享特征层
        self.shared_layers = nn.Sequential(
            nn.Linear(total_input_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # 策略头 (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            # 移除Softmax，在forward中使用log_softmax
        )
        
        # 价值头 (critic)
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 使用正交初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    def forward(self, x: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播，返回策略分布、对数概率和状态价值"""
        # 展平所有输入
        map_flat = x['map'].flatten(start_dim=1)
        tanks_flat = x['tanks'].flatten(start_dim=1)
        bullets_flat = x['bullets'].flatten(start_dim=1)
        
        # 拼接所有特征
        combined = torch.cat([map_flat, tanks_flat, bullets_flat], dim=1)
        
        # 共享特征提取
        shared_features = self.shared_layers(combined)
        
        # 策略和价值输出
        logits = self.policy_head(shared_features)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        
        value = self.value_head(shared_features)
        
        return probs, log_probs, value


class PPOAgent(BaseAgent):
    """PPO智能体，支持异步训练"""
    
    def __init__(self, state_shape: Dict, action_dim: int, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.device = device
        
        # 创建网络
        self.network = PPONetwork(state_shape, action_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-4, eps=1e-5)  # 调整学习率和epsilon
        
        # PPO超参数
        self.gamma = 0.99  # 折扣因子
        self.gae_lambda = 0.95  # GAE参数
        self.clip_epsilon = 0.2  # PPO截断参数
        self.entropy_coef = 0.03  # 熵正则化系数
        self.value_coef = 0.1  # 价值损失系数
        self.max_grad_norm = 0.5  # 梯度截断
        
        # 经验缓冲区
        self.buffer = PPOBuffer()
        self.replay_buffer = self.buffer  # 为了兼容性，添加replay_buffer别名
        
        # 训练相关
        self.train_step = 0
        
    def select_action(self, state: Dict, training: bool = True) -> int:
        """选择动作"""
        with torch.no_grad():            
            state_tensor = self._dict_to_tensor(state)
            policy, log_probs, value = self.network(state_tensor)
            
            if training:
                # 根据策略分布采样
                action_dist = torch.distributions.Categorical(policy)
                action = action_dist.sample()
                log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
                
                # 存储选择信息用于训练
                self.last_action_info = {
                    'action': action.item(),
                    'log_prob': log_prob.item(),
                    'value': value.item(),
                    'policy': policy.cpu().numpy()
                }
                
                return action.item()
            else:
                # 评估模式，选择最可能的动作
                return torch.argmax(policy, dim=-1).item()
    
    def store_transition(self, state: Dict, action: int, reward: float, next_state: Dict, done: bool, info: Dict = None):
        """存储转换"""
        self.buffer.store(state, action, reward, next_state, done, info)
    
    def train(self, batch_size: int = 64, epochs: int = 4):
        """训练网络"""
        if len(self.buffer) < batch_size:
            return None
            
        total_loss = 0
        num_updates = 0
        
        # 计算优势和回报
        self.buffer.compute_advantages_and_returns(self.gamma, self.gae_lambda)
        
        # 多轮训练
        for _ in range(epochs):
            for batch in self.buffer.get_batches(batch_size):
                loss = self._update_network(batch)
                total_loss += loss
                num_updates += 1
                
        self.buffer.clear()
        self.train_step += 1
        print(f"训练步骤: {self.train_step}, 平均损失: {total_loss / max(num_updates, 1):.4f}")
        return total_loss / max(num_updates, 1)
    
    def _update_network(self, batch: Dict) -> float:
        """更新网络参数"""
        states = self._batch_dict_to_tensor(batch['states'])
        actions = torch.tensor(batch['actions'], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(batch['log_probs'], dtype=torch.float32, device=self.device)
        returns = torch.tensor(batch['returns'], dtype=torch.float32, device=self.device)
        advantages = torch.tensor(batch['advantages'], dtype=torch.float32, device=self.device)
        
        # 标准化优势，添加检查以避免零方差
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > 0:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
          # 前向传播
        policy, log_probs, values = self.network(states)
        values = values.squeeze(-1)  # 确保维度匹配
        
        # 计算策略损失
        try:
            # 获取对应动作的对数概率
            action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            
            # 直接计算比率
            ratio = torch.exp(action_log_probs - old_log_probs)
            
            # PPO裁剪目标
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 确保返回维度匹配
            returns = returns.view_as(values)
            value_loss = F.mse_loss(values, returns)
              # 计算熵损失
            action_dist = torch.distributions.Categorical(probs=policy)
            entropy = action_dist.entropy().mean()
            entropy_loss = -self.entropy_coef * entropy
            
        except Exception as e:
            print(f"警告：策略计算出错: {e}")
            return 0.0
        
        # 总损失
        total_loss = policy_loss + self.value_coef * value_loss + entropy_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return total_loss.item()
    
    def _dict_to_tensor(self, state: Dict) -> Dict:
        """将状态字典转换为张量字典"""
        expected_map_shape = self.state_shape['map']
        expected_tanks_shape = self.state_shape['tanks']
        expected_bullets_shape = self.state_shape['bullets']
        
        tank_feature_keys = ['x', 'y', 'angle', 'hp']
        bullet_feature_keys = ['x', 'y', 'angle']
        
        # 处理地图
        map_np = np.zeros(expected_map_shape, dtype=np.float32)
        raw_map_data = state.get('map')
        if raw_map_data is not None:
            try:
                data_to_assign = np.asarray(raw_map_data, dtype=np.float32)
                slices_source = tuple(slice(min(s_dim, e_dim)) for s_dim, e_dim in zip(data_to_assign.shape, expected_map_shape))
                slices_target = tuple(slice(min(s_dim, e_dim)) for s_dim, e_dim in zip(data_to_assign.shape, expected_map_shape))
                map_np[slices_target] = data_to_assign[slices_source]
            except Exception:
                pass
        
        # 处理坦克
        tanks_np = np.zeros(expected_tanks_shape, dtype=np.float32)
        raw_tanks_data = state.get('tanks', [])
        if isinstance(raw_tanks_data, list):
            for i, tank_data_dict in enumerate(raw_tanks_data):
                if i < expected_tanks_shape[0] and isinstance(tank_data_dict, dict):
                    for j, key in enumerate(tank_feature_keys):
                        if j < expected_tanks_shape[1]:
                            tanks_np[i, j] = float(tank_data_dict.get(key, 0.0))
        
        # 处理子弹
        bullets_np = np.zeros(expected_bullets_shape, dtype=np.float32)
        raw_bullets_data = state.get('bullets', [])
        if isinstance(raw_bullets_data, list):
            for i, bullet_data_dict in enumerate(raw_bullets_data):
                if i < expected_bullets_shape[0] and isinstance(bullet_data_dict, dict):
                    for j, key in enumerate(bullet_feature_keys):
                        if j < expected_bullets_shape[1]:
                            bullets_np[i, j] = float(bullet_data_dict.get(key, 0.0))
        
        return {
            'map': torch.from_numpy(map_np).to(self.device).unsqueeze(0),
            'tanks': torch.from_numpy(tanks_np).to(self.device).unsqueeze(0),
            'bullets': torch.from_numpy(bullets_np).to(self.device).unsqueeze(0)
        }
    
    def _batch_dict_to_tensor(self, states: List[Dict]) -> Dict:
        """批量转换状态字典为张量"""
        expected_map_shape = self.state_shape['map']
        expected_tanks_shape = self.state_shape['tanks']
        expected_bullets_shape = self.state_shape['bullets']
        
        tank_feature_keys = ['x', 'y', 'angle', 'hp']
        bullet_feature_keys = ['x', 'y', 'angle']
        
        maps_list = []
        tanks_list = []
        bullets_list = []
        
        for s in states:
            # 处理地图
            current_map_arr = np.zeros(expected_map_shape, dtype=np.float32)
            raw_map_data = s.get('map')
            if raw_map_data is not None:
                try:
                    data_np = np.asarray(raw_map_data, dtype=np.float32)
                    if data_np.ndim == len(expected_map_shape):
                        slice_dims = [min(data_np.shape[i], expected_map_shape[i]) for i in range(data_np.ndim)]
                        current_map_arr[tuple(slice(sd) for sd in slice_dims)] = data_np[tuple(slice(sd) for sd in slice_dims)]
                except Exception:
                    pass
            maps_list.append(current_map_arr)
            
            # 处理坦克
            processed_tank_features = np.zeros(expected_tanks_shape, dtype=np.float32)
            raw_tanks_data = s.get('tanks', [])
            if isinstance(raw_tanks_data, list):
                for i in range(expected_tanks_shape[0]):
                    if i < len(raw_tanks_data) and isinstance(raw_tanks_data[i], dict):
                        tank_dict = raw_tanks_data[i]
                        for j, key in enumerate(tank_feature_keys):
                            if j < expected_tanks_shape[1]:
                                processed_tank_features[i, j] = float(tank_dict.get(key, 0.0))
            tanks_list.append(processed_tank_features)
            
            # 处理子弹
            processed_bullet_features = np.zeros(expected_bullets_shape, dtype=np.float32)
            raw_bullets_data = s.get('bullets', [])
            if isinstance(raw_bullets_data, list):
                for i in range(expected_bullets_shape[0]):
                    if i < len(raw_bullets_data) and isinstance(raw_bullets_data[i], dict):
                        bullet_dict = raw_bullets_data[i]
                        for j, key in enumerate(bullet_feature_keys):
                            if j < expected_bullets_shape[1]:
                                processed_bullet_features[i, j] = float(bullet_dict.get(key, 0.0))
            bullets_list.append(processed_bullet_features)
        
        maps_np = np.stack(maps_list)
        tanks_np = np.stack(tanks_list)
        bullets_np = np.stack(bullets_list)
        
        return {
            'map': torch.from_numpy(maps_np).to(self.device),
            'tanks': torch.from_numpy(tanks_np).to(self.device),
            'bullets': torch.from_numpy(bullets_np).to(self.device)
        }
    
    def get_state_dict(self) -> dict:
        """获取网络状态字典"""
        return self.network.state_dict()
    
    def load_state_dict(self, state_dict: dict):
        """加载网络状态字典"""
        self.network.load_state_dict(state_dict)
    
    def save_checkpoint(self, path: str):
        """保存完整检查点"""
        checkpoint = {
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_step': self.train_step,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_epsilon': self.clip_epsilon,
            'entropy_coef': self.entropy_coef,
            'value_coef': self.value_coef
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """加载完整检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_step = checkpoint.get('train_step', 0)
        self.gamma = checkpoint.get('gamma', self.gamma)
        self.gae_lambda = checkpoint.get('gae_lambda', self.gae_lambda)
        self.clip_epsilon = checkpoint.get('clip_epsilon', self.clip_epsilon)
        self.entropy_coef = checkpoint.get('entropy_coef', self.entropy_coef)
        self.value_coef = checkpoint.get('value_coef', self.value_coef)
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_step': self.train_step
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_step = checkpoint['train_step']


class PPOBuffer:
    """PPO经验回放缓冲区"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.advantages = []
        self.returns = []
        
    def push(self, state: Dict, action: int, reward: float, next_state: Dict, done: bool):
        """兼容性方法：存储经验（与DQN接口兼容）"""
        self.store(state, action, reward, next_state, done)
    
    def store(self, state: Dict, action: int, reward: float, next_state: Dict, done: bool, info: Dict = None):
        """存储经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        
        # 存储动作信息(如果有)
        if info and 'log_prob' in info:
            self.log_probs.append(info['log_prob'])
        if info and 'value' in info:
            self.values.append(info['value'])
    
    def compute_advantages_and_returns(self, gamma: float, gae_lambda: float):
        """计算优势和回报"""
        # 如果没有价值估计，使用0
        if not self.values:
            self.values = [0.0] * len(self.rewards)
        
        advantages = []
        returns = []
        
        # 计算GAE优势
        gae = 0
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[i + 1]
            
            delta = self.rewards[i] + gamma * next_value * (1 - self.dones[i]) - self.values[i]
            gae = delta + gamma * gae_lambda * (1 - self.dones[i]) * gae
            advantages.insert(0, gae)
        
        # 计算回报
        for i in range(len(advantages)):
            returns.append(advantages[i] + self.values[i])
        
        self.advantages = advantages
        self.returns = returns
    
    def get_batches(self, batch_size: int):
        """获取训练批次"""
        indices = list(range(len(self.states)))
        np.random.shuffle(indices)
        
        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            batch = {
                'states': [self.states[i] for i in batch_indices],
                'actions': [self.actions[i] for i in batch_indices],
                'log_probs': [self.log_probs[i] if i < len(self.log_probs) else 0.0 for i in batch_indices],
                'returns': [self.returns[i] for i in batch_indices],
                'advantages': [self.advantages[i] for i in batch_indices]
            }
            
            yield batch
    
    def clear(self):
        """清空缓冲区"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.advantages.clear()
        self.returns.clear()
    
    def __len__(self):
        return len(self.states)
