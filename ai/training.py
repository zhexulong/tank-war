import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from ai.environment import TankBattleEnv
from ai.agent import DQNAgent, MultiAgentDQN
from game.config import RL_SETTINGS

def train_single_agent(episodes: int = 1000, save_interval: int = 100, render: bool = False):
    """训练单个智能体"""
    # 创建环境
    env = TankBattleEnv(render_mode='human' if render else None)
    
    # 获取状态和动作空间
    state_shape = {
        'map': env.observation_space['map'].shape,
        'tanks': env.observation_space['tanks'].shape,
        'bullets': env.observation_space['bullets'].shape
    }
    action_dim = env.action_space.n
    
    # 创建智能体
    agent = DQNAgent(state_shape, action_dim)
    
    # 创建保存目录
    os.makedirs('models', exist_ok=True)
    
    # 训练记录
    rewards = []
    losses = []
    win_rates = []
    episode_lengths = []
    
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
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # 训练智能体
            loss = agent.train()
            if loss is not None:
                episode_loss += loss
            
            # 更新状态和奖励
            state = next_state
            episode_reward += reward
            episode_step += 1
            
            # 渲染
            if render:
                env.render()
                time.sleep(0.01)
        
        # 记录训练数据
        rewards.append(episode_reward)
        losses.append(episode_loss / max(1, episode_step))
        win = info.get('winner', 0) == 1
        win_rates.append(1 if win else 0)
        episode_lengths.append(episode_step)
        
        # 打印训练信息
        print(f"Episode {episode}/{episodes}, Reward: {episode_reward:.2f}, Loss: {episode_loss/max(1, episode_step):.4f}, "  
              f"Win: {win}, Steps: {episode_step}, Epsilon: {agent.epsilon:.4f}")
        
        # 保存模型
        if episode % save_interval == 0:
            agent.save(f"models/agent_episode_{episode}.pt")
            
            # 绘制训练曲线
            plot_training_curves(rewards, losses, win_rates, episode_lengths, episode)
    
    # 保存最终模型
    agent.save("models/agent_final.pt")
    
    # 关闭环境
    env.close()
    
    # 绘制最终训练曲线
    plot_training_curves(rewards, losses, win_rates, episode_lengths, episodes)
    
    return agent

def train_multi_agent(episodes: int = 1000, save_interval: int = 100, render: bool = False):
    """训练多智能体"""
    # 创建环境
    env = TankBattleEnv(render_mode='human' if render else None)
    
    # 获取状态和动作空间
    state_shape = {
        'map': env.observation_space['map'].shape,
        'tanks': env.observation_space['tanks'].shape,
        'bullets': env.observation_space['bullets'].shape
    }
    action_dim = env.action_space.n
    
    # 创建多智能体
    multi_agent = MultiAgentDQN(state_shape, action_dim, num_agents=2)
    
    # 创建保存目录
    os.makedirs('models', exist_ok=True)
    
    # 训练记录
    rewards = [[], []]
    win_rates = [[], []]
    
    # 训练循环
    for episode in range(1, episodes + 1):
        # 重置环境
        state = env.reset()
        episode_rewards = [0, 0]
        done = False
        
        # 获取两个智能体的状态
        states = [state, state]  # 简化处理，实际应该为每个智能体提供不同视角的状态
        
        # 单局游戏循环
        while not done:
            # 选择动作
            actions = multi_agent.select_actions(states)
            
            # 执行动作（这里简化为只执行第一个智能体的动作）
            next_state, reward, done, info = env.step(actions[0])
            
            # 更新状态和奖励
            states = [next_state, next_state]  # 简化处理
            episode_rewards[0] += reward
            episode_rewards[1] += -reward  # 简化处理，假设是零和游戏
            
            # 存储经验并训练（简化处理）
            for i in range(2):
                agent = multi_agent.agents[i]
                agent.replay_buffer.push(states[i], actions[i], episode_rewards[i], states[i], done)
            
            # 训练智能体
            multi_agent.train()
            
            # 渲染
            if render:
                env.render()
                time.sleep(0.01)
        
        # 记录训练数据
        for i in range(2):
            rewards[i].append(episode_rewards[i])
            win = info.get('winner', 0) == i + 1
            win_rates[i].append(1 if win else 0)
        
        # 打印训练信息
        print(f"Episode {episode}/{episodes}, Rewards: {episode_rewards}, "  
              f"Winner: {info.get('winner', 0)}, Epsilon: {multi_agent.agents[0].epsilon:.4f}")
        
        # 保存模型
        if episode % save_interval == 0:
            multi_agent.save([f"models/agent1_episode_{episode}.pt", f"models/agent2_episode_{episode}.pt"])
    
    # 保存最终模型
    multi_agent.save(["models/agent1_final.pt", "models/agent2_final.pt"])
    
    # 关闭环境
    env.close()
    
    return multi_agent

def plot_training_curves(
    rewards: List[float], losses: List[float], win_rates: List[int], episode_lengths: List[int],
    moving_avg_reward: List[float] = None, moving_avg_loss: List[float] = None, moving_avg_win_rate: List[float] = None,
    episode: int = None
):
    """绘制训练曲线，包括原始数据和移动平均线"""
    plt.figure(figsize=(15, 10))
    
    # 绘制奖励曲线
    plt.subplot(2, 2, 1)
    plt.plot(rewards, alpha=0.5, label='Raw')
    if moving_avg_reward:
        plt.plot(range(len(rewards)-len(moving_avg_reward), len(rewards)), moving_avg_reward, 'r', label='Moving Average')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    # 绘制损失曲线
    plt.subplot(2, 2, 2)
    plt.plot(losses, alpha=0.5, label='Raw')
    if moving_avg_loss:
        plt.plot(range(len(losses)-len(moving_avg_loss), len(losses)), moving_avg_loss, 'r', label='Moving Average')
    plt.title('Average Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制胜率曲线
    plt.subplot(2, 2, 3)
    win_rate_avg = [np.mean(win_rates[max(0, i-100):i+1]) for i in range(len(win_rates))]
    plt.plot(win_rate_avg)
    plt.title('Win Rate (100-episode moving average)')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.ylim(0, 1)
    
    # 绘制回合长度曲线
    plt.subplot(2, 2, 4)
    plt.plot(episode_lengths)
    plt.title('Episode Length')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.tight_layout()
    plt.savefig(f"training_curves_episode_{episode}.png")
    plt.close()

def evaluate_agent(agent_path: str, episodes: int = 10, render: bool = True):
    """评估智能体性能"""
    # 创建环境
    env = TankBattleEnv(render_mode='human' if render else None)
    
    # 获取状态和动作空间
    state_shape = {
        'map': env.observation_space['map'].shape,
        'tanks': env.observation_space['tanks'].shape,
        'bullets': env.observation_space['bullets'].shape
    }
    action_dim = env.action_space.n
    
    # 创建智能体并加载模型
    agent = DQNAgent(state_shape, action_dim)
    agent.load(agent_path)
    agent.epsilon = 0.0  # 关闭探索
    
    # 评估记录
    rewards = []
    win_count = 0
    
    # 评估循环
    for episode in range(1, episodes + 1):
        # 重置环境
        state = env.reset()
        episode_reward = 0
        done = False
        
        # 单局游戏循环
        while not done:
            # 选择动作
            action = agent.select_action(state, training=False)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 更新状态和奖励
            state = next_state
            episode_reward += reward
            
            # 渲染
            if render:
                env.render()
                time.sleep(0.05)
        
        # 记录评估数据
        rewards.append(episode_reward)
        win = info.get('winner', 0) == 1
        if win:
            win_count += 1
        
        # 打印评估信息
        print(f"Episode {episode}/{episodes}, Reward: {episode_reward:.2f}, Win: {win}")
    
    # 打印总体评估结果
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {np.mean(rewards):.2f}")
    print(f"Win Rate: {win_count/episodes:.2f}")
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    # 训练单个智能体
    train_single_agent(episodes=5000, save_interval=100, render=False)
    
    # 评估智能体
    # evaluate_agent("models/agent_final.pt", episodes=10, render=True)
    
    # 训练多智能体
    # train_multi_agent(episodes=500, save_interval=50, render=True)