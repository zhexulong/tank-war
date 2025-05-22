import os
import time
import numpy as np
from typing import Dict, List, Tuple
from ai.base_agent import BaseAgent
from ai.agent import DQNAgent
from ai.logic_agent import LogicAgent
from ai.environment import TankBattleEnv
from ai.simplified_env import SimplifiedGameEnv
from ai.training import plot_training_curves

def train_against_logic(episodes: int = 1000, save_interval: int = 100, render: bool = False, checkpoint_path: str = None):
    """训练RL智能体对抗Logic智能体"""
    # 创建环境
    env = SimplifiedGameEnv(render_mode='human' if render else None)
    state_shape = {
        'map': env.observation_space['map'],
        'tanks': env.observation_space['tanks'],
        'bullets': env.observation_space['bullets']
    }
    action_dim = env.action_space.n
    rl_agent = DQNAgent(state_shape, action_dim)
    logic_agent = LogicAgent()
    
    # 创建保存目录
    os.makedirs('models', exist_ok=True)
    
    # 训练记录
    rewards = []
    losses = []
    win_rates = []
    episode_lengths = []
    
    # 加载检查点（如果存在）
    start_episode = 1
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        rl_agent.load_state_dict(checkpoint['model_state_dict'])
        rl_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        rewards = checkpoint['rewards']
        losses = checkpoint['losses']
        win_rates = checkpoint['win_rates']
        episode_lengths = checkpoint['episode_lengths']
        start_episode = checkpoint['episode'] + 1
        print(f"加载检查点：{checkpoint_path}，从第{start_episode}轮继续训练")
    
    # 计算移动平均
    window_size = 100
    moving_avg_reward = []
    moving_avg_loss = []
    moving_avg_win_rate = []
    
    # 训练循环
    for episode in range(1, episodes + 1):
        # 重置环境
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        episode_step = 0
        done = False
        
        # 单局游戏循环
        while not done:
            # RL智能体选择动作
            rl_action = rl_agent.select_action(state)
            
            # Logic智能体选择动作
            logic_action = logic_agent.select_action(state)
            
            # 执行动作
            next_state, rewards, done, info = env.step([rl_action, logic_action])
            
            # 存储RL智能体的经验
            rl_agent.replay_buffer.push(state, rl_action, rewards[0], next_state, done)
            
            # 训练RL智能体
            loss = rl_agent.train()
            if loss is not None:
                episode_loss += loss
            
            # 更新状态和奖励
            state = next_state
            episode_reward += rewards[0]  # 只记录RL智能体的奖励
            episode_step += 1
            
            # 渲染
            if render:
                env.render()
                time.sleep(0.01)
        
        # 记录训练数据
        rewards.append(episode_reward)
        losses.append(episode_loss / max(1, episode_step))
        win = info.get('winner', 0) == 0  # RL智能体是玩家0
        win_rates.append(1 if win else 0)
        episode_lengths.append(episode_step)
        
        # 打印训练信息
        print(f"Episode {episode}/{episodes}, Reward: {episode_reward:.2f}, Loss: {episode_loss/max(1, episode_step):.4f}, "  
              f"Win: {win}, Steps: {episode_step}, Epsilon: {rl_agent.epsilon:.4f}")
        
        # 计算移动平均
        if len(rewards) >= window_size:
            moving_avg_reward.append(np.mean(rewards[-window_size:]))
            moving_avg_loss.append(np.mean(losses[-window_size:]))
            moving_avg_win_rate.append(np.mean(win_rates[-window_size:]))
        
        # 保存检查点和模型
        if episode % save_interval == 0:
            # 保存检查点
            checkpoint = {
                'episode': episode,
                'model_state_dict': rl_agent.state_dict(),
                'optimizer_state_dict': rl_agent.optimizer.state_dict(),
                'rewards': rewards,
                'losses': losses,
                'win_rates': win_rates,
                'episode_lengths': episode_lengths,
                'moving_avg_reward': moving_avg_reward,
                'moving_avg_loss': moving_avg_loss,
                'moving_avg_win_rate': moving_avg_win_rate
            }
            torch.save(checkpoint, f"models/rl_vs_logic_checkpoint_{episode}.pt")
            
            # 保存模型
            rl_agent.save(f"models/rl_vs_logic_episode_{episode}.pt")
            
            # 绘制训练曲线
            plot_training_curves(
                rewards, losses, win_rates, episode_lengths,
                moving_avg_reward, moving_avg_loss, moving_avg_win_rate,
                episode
            )
    
    # 保存最终模型
    rl_agent.save("models/rl_vs_logic_final.pt")
    
    # 关闭环境
    env.close()
    
    # 绘制最终训练曲线
    plot_training_curves(rewards, losses, win_rates, episode_lengths, episodes)
    
    return rl_agent

def evaluate_agents(rl_agent: BaseAgent, logic_agent: BaseAgent, num_episodes: int = 100, render: bool = False) -> Tuple[float, float]:
    """评估两个智能体的对抗性能
    
    Args:
        rl_agent: RL智能体
        logic_agent: Logic智能体
        num_episodes: 评估局数
        render: 是否渲染游戏画面
        
    Returns:
        win_rate: RL智能体的胜率
        avg_reward: RL智能体的平均奖励
    """
    env = TankBattleEnv(render_mode='human' if render else None)
    
    total_wins = 0
    total_reward = 0
    
    for episode in range(num_episodes):
        states = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 选择动作
            rl_action = rl_agent.select_action(states[0], training=False)
            logic_action = logic_agent.select_action(states[1], training=False)
            
            # 执行动作
            states, rewards, done, info = env.step([rl_action, logic_action])
            
            # 更新奖励
            episode_reward += rewards[0]
            
            # 渲染
            if render:
                env.render()
                time.sleep(0.01)
        
        # 记录胜利
        if info.get('winner', 0) == 0:  # RL智能体是玩家0
            total_wins += 1
        
        total_reward += episode_reward
        
        # 打印评估信息
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Win: {info.get('winner', 0) == 0}")
    
    env.close()
    
    win_rate = total_wins / num_episodes
    avg_reward = total_reward / num_episodes
    
    print(f"\nEvaluation Results:")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Reward: {avg_reward:.2f}")
    
    return win_rate, avg_reward