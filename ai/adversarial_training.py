import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from ai.base_agent import BaseAgent
from ai.agent import DQNAgent
from ai.logic_agent import LogicAgent, LogicExpertAgent
from ai.environment import TankBattleEnv
from ai.simplified_env import SimplifiedGameEnv
from ai.training import plot_training_curves, plot_expert_performance_relation
import torch

def _action_to_str_dqn(action: int) -> str:
    """DQN动作ID转字符串描述
    
    Args:
        action: DQN智能体的动作ID
        
    Returns:
        动作的文字描述
    """
    action_map = {
        0: "停留",
        1: "前进", 
        2: "后退", 
        3: "左转", 
        4: "右转", 
        5: "开火"
    }
    return action_map.get(action, "未知")

def _action_to_str_logic(action: int) -> str:
    """逻辑智能体动作ID转字符串描述
    
    Args:
        action: 逻辑智能体的动作ID
        
    Returns:
        动作的文字描述
    """
    action_map = {
        0: "停留",
        1: "前进",
        2: "左转",
        3: "右转",
        4: "开火"
    }
    return action_map.get(action, "未知")


def _calculate_expert_reward(episode: int, total_episodes: int, initial_reward: float, decay_factor: float) -> float:
    """计算当前回合的专家奖励值（考虑线性衰减）
    
    Args:
        episode: 当前训练回合
        total_episodes: 总训练回合数
        initial_reward: 初始专家奖励值
        decay_factor: 专家奖励衰减因子 (1.0表示完全按照训练进度线性衰减，0表示不衰减)
        
    Returns:
        当前回合的专家奖励值
    """
    # 计算训练进度比例 (0.0-1.0)
    progress_ratio = min(1.0, (episode - 1) / total_episodes)
    
    # 根据衰减因子计算当前奖励值
    current_reward = initial_reward * max(0.0, 1.0 - progress_ratio * decay_factor)
    
    return current_reward

def train_against_logic(
    episodes: int = 1000, 
    save_interval: int = 100, 
    render: bool = False, 
    checkpoint_path: str = None,
    use_expert_guidance: bool = False,
    expert_reward_init: float = 100.0,
    expert_decay_factor: float = 1.0
):
    """训练RL智能体对抗Logic智能体
    
    Args:
        episodes: 训练回合数
        save_interval: 模型保存间隔
        render: 是否渲染游戏画面
        checkpoint_path: 检查点路径，如果存在则从检查点继续训练
        use_expert_guidance: 是否使用专家策略学习
        expert_reward_init: 初始专家奖励值
        expert_decay_factor: 专家奖励衰减因子 (1.0表示完全按照训练进度线性衰减)
    """
    # 创建环境
    env = SimplifiedGameEnv(render_mode='human' if render else None)
    state_shape = {
        'map': env.observation_space['map'],
        'tanks': env.observation_space['tanks'],
        'bullets': env.observation_space['bullets']
    }
    action_dim = env.action_space.n
    rl_agent = DQNAgent(state_shape, action_dim)
    logic_agent = LogicAgent()  # 作为对手智能体
    expert_agent = LogicExpertAgent()  # 作为专家智能体
    
    # 创建保存目录
    os.makedirs('models', exist_ok=True)
    
    # 训练记录
    rewards = []
    losses = []
    win_rates = []
    episode_lengths = []
    expert_agreements = []  # 记录与专家动作一致的比例
    
    # 移动平均记录
    window_size = 100
    moving_avg_reward = []
    moving_avg_loss = []
    moving_avg_win_rate = []
    moving_avg_expert_agreement = []  # 专家一致性的移动平均
    
    # 已在上方定义移动平均列表
    
    # 加载检查点（如果存在）
    start_episode = 1
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        rl_agent.q_network.load_state_dict(checkpoint['model_state_dict'])
        rl_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        rewards = checkpoint['rewards']
        losses = checkpoint['losses']
        win_rates = checkpoint['win_rates']
        episode_lengths = checkpoint['episode_lengths']
        expert_agreements = checkpoint.get('expert_agreements', [])
        moving_avg_reward = checkpoint.get('moving_avg_reward', [])
        moving_avg_loss = checkpoint.get('moving_avg_loss', [])
        moving_avg_win_rate = checkpoint.get('moving_avg_win_rate', [])
        moving_avg_expert_agreement = checkpoint.get('moving_avg_expert_agreement', [])
        
        # 可选地从检查点恢复专家学习设置
        if 'use_expert_guidance' in checkpoint and 'expert_reward_init' in checkpoint:
            use_expert_guidance = checkpoint.get('use_expert_guidance', use_expert_guidance)
            expert_reward_init = checkpoint.get('expert_reward_init', expert_reward_init)
            expert_decay_factor = checkpoint.get('expert_decay_factor', expert_decay_factor)
            
        start_episode = checkpoint['episode'] + 1
        print(f"加载检查点：{checkpoint_path}，从第{start_episode}轮继续训练")
    
    # 训练循环
    for episode in range(start_episode, episodes + 1):
        # 重置环境
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        episode_step = 0
        expert_agreement_count = 0  # 记录本回合与专家动作一致的次数
        done = False
        
        # 单局游戏循环
        while not done:
            # RL智能体选择动作（player_id=0）
            rl_action = rl_agent.select_action(state)
            
            # 专家智能体从RL智能体视角选择动作（提供专家建议）
            expert_action = expert_agent.select_action(state)
            
            # 对手Logic智能体选择动作（player_id=1，用于实际游戏）
            logic_action = logic_agent.select_action(state)
            
            # 记录决策过程（用于调试）
            if episode % 100 == 0 and episode_step < 5:  # 每100回合的前5步骤记录详情
                print(f"\n[回合 {episode}, 步骤 {episode_step}] 决策详情:")
                print(f"- RL智能体选择动作: {rl_action} ({_action_to_str_dqn(rl_action)})")
                print(f"- 专家建议动作: {expert_action} ({_action_to_str_logic(expert_action)})")
                print(f"- 对手动作: {logic_action} ({_action_to_str_logic(logic_action)})")
            
            # 应用专家策略学习
            current_expert_reward = 0.0
            if use_expert_guidance:
                # 计算当前专家奖励值（根据训练进度衰减）
                current_expert_reward = _calculate_expert_reward(
                    episode=episode, 
                    total_episodes=episodes,
                    initial_reward=expert_reward_init,
                    decay_factor=expert_decay_factor
                )
                
                # 动作空间映射：
                # DQNAgent: 0-stay, 1-forward, 2-backward, 3-left, 4-right, 5-fire
                # LogicAgent: 0-stay, 1-forward, 2-turn_left, 3-turn_right, 4-fire
                
                # 将RL动作映射到专家动作空间进行比较
                matched = False
                if rl_action == 0 and expert_action == 0:  # stay = stay
                    matched = True
                elif rl_action == 1 and expert_action == 1:  # forward = forward
                    matched = True
                elif rl_action == 3 and expert_action == 2:  # left = turn_left
                    matched = True
                elif rl_action == 4 and expert_action == 3:  # right = turn_right
                    matched = True
                elif rl_action == 5 and expert_action == 4:  # fire = fire
                    matched = True
                
                # 后退动作(2)在专家动作空间中不存在，永远不会匹配
                
                # 检查RL智能体的动作是否与专家建议一致
                if matched:
                    expert_agreement_count += 1  # 记录一致性
                else:
                    current_expert_reward = 0.0  # 不一致时不给奖励
                    
                if episode % 100 == 0 and episode_step < 5:  # 每100回合的前5步骤记录详情
                    print(f"- 动作一致性: {'匹配' if matched else '不匹配'}")
                    print(f"- 专家奖励: {current_expert_reward:.4f}")
            
            # 执行动作
            next_state, step_rewards, done, info = env.step([rl_action, logic_action])
            
            # 应用专家奖励（如果有）
            step_rewards = list(step_rewards)  # 转为列表以便修改
            step_rewards[0] += current_expert_reward  # 添加专家策略奖励
            
            # 存储RL智能体的经验
            rl_agent.replay_buffer.push(state, rl_action, step_rewards[0], next_state, done)
            
            # 训练RL智能体
            loss = rl_agent.train()
            if loss is not None:
                episode_loss += loss
            
            # 更新状态和奖励
            state = next_state
            episode_reward += step_rewards[0]  # 只记录RL智能体的奖励（包含专家策略奖励）
            episode_step += 1
            
            # 渲染
            if render:
                env.render()
                time.sleep(0.01)
        
        # 记录训练数据
        rewards.append(episode_reward)
        losses.append(episode_loss / max(1, episode_step))
        win = info.get('winner', 0) == 1  # RL智能体是玩家1
        win_rates.append(1 if win else 0)
        episode_lengths.append(episode_step)
        
        # 计算与专家动作的一致性
        expert_agreement_rate = expert_agreement_count / max(1, episode_step)
        expert_agreements.append(expert_agreement_rate)
        
        # 打印训练信息
        expert_info = ""
        if use_expert_guidance:
            progress_ratio = min(1.0, (episode - 1) / episodes)
            current_expert_reward = expert_reward_init * max(0.0, 1.0 - progress_ratio * expert_decay_factor)
            expert_agreement_rate = expert_agreement_count / max(1, episode_step)
            expert_info = f", Expert reward: {current_expert_reward:.4f}, Agreement: {expert_agreement_rate:.2%}"
            
        print(f"Episode {episode}/{episodes}, Reward: {episode_reward:.2f}, Loss: {episode_loss/max(1, episode_step):.4f}, "  
              f"Win: {win}, Steps: {episode_step}, Epsilon: {rl_agent.epsilon:.4f}{expert_info}")
        
        # 计算移动平均（使用可用的所有数据，最多window_size个）
        current_window_size = min(len(rewards), window_size)
        if current_window_size > 0:
            moving_avg_reward.append(np.mean(rewards[-current_window_size:]))
            moving_avg_loss.append(np.mean(losses[-current_window_size:]))
            moving_avg_win_rate.append(np.mean(win_rates[-current_window_size:]))
            moving_avg_expert_agreement.append(np.mean(expert_agreements[-current_window_size:]))
        
        # 保存检查点和模型
        if episode % save_interval == 0:
            # 保存检查点
            checkpoint = {
                'episode': episode,
                'model_state_dict': rl_agent.q_network.state_dict(),
                'optimizer_state_dict': rl_agent.optimizer.state_dict(),
                'rewards': rewards,
                'losses': losses,
                'win_rates': win_rates,
                'episode_lengths': episode_lengths,
                'expert_agreements': expert_agreements,
                'moving_avg_reward': moving_avg_reward,
                'moving_avg_loss': moving_avg_loss,
                'moving_avg_win_rate': moving_avg_win_rate,
                'moving_avg_expert_agreement': moving_avg_expert_agreement,
                # 保存专家策略学习的配置
                'use_expert_guidance': use_expert_guidance,
                'expert_reward_init': expert_reward_init,
                'expert_decay_factor': expert_decay_factor
            }
            torch.save(checkpoint, f"models/rl_vs_logic_checkpoint_{episode}.pt")
            
            # 保存模型
            rl_agent.save(f"models/rl_vs_logic_episode_{episode}.pt")
            
            # 绘制训练曲线
            plot_training_curves(
                rewards, losses, win_rates, episode_lengths,
                moving_avg_reward=moving_avg_reward.copy(), 
                moving_avg_loss=moving_avg_loss.copy(), 
                moving_avg_win_rate=moving_avg_win_rate.copy(),
                episode=episode,
                expert_agreements=expert_agreements.copy(),
                moving_avg_expert_agreement=moving_avg_expert_agreement.copy(),
                use_expert_guidance=use_expert_guidance
            )
            
            # 如果启用了专家策略学习，额外绘制专家一致率与性能关系图
            if use_expert_guidance and len(expert_agreements) > 0:
                plot_expert_performance_relation(
                    rewards, win_rates, expert_agreements,
                    episode=episode
                )
    
    # 保存最终模型
    rl_agent.save("models/rl_vs_logic_final.pt")
    
    # 关闭环境
    env.close()
    # 输出最终训练结果摘要
    if len(moving_avg_reward) > 0:
        print(f"\n===== 训练结果摘要 =====")
        print(f"最终移动平均奖励: {moving_avg_reward[-1]:.4f}")
        print(f"最终移动平均损失: {moving_avg_loss[-1]:.4f}")
        print(f"最终移动平均胜率: {moving_avg_win_rate[-1]:.2%}")
    
    # 如果启用了专家策略学习，分析专家一致率的变化
    if use_expert_guidance and len(expert_agreements) > 0:
        # 计算不同训练阶段的专家一致率
        early_stage = int(len(expert_agreements) * 0.25)
        mid_stage = int(len(expert_agreements) * 0.5)
        late_stage = int(len(expert_agreements) * 0.75)
        
        avg_early = np.mean(expert_agreements[:early_stage])
        avg_mid = np.mean(expert_agreements[early_stage:mid_stage])
        avg_late_mid = np.mean(expert_agreements[mid_stage:late_stage])
        avg_late = np.mean(expert_agreements[late_stage:])
        
        print(f"\n===== 专家策略学习分析 =====")
        print(f"初期专家一致率 (0-25%): {avg_early:.2%}")
        print(f"中期专家一致率 (25-50%): {avg_mid:.2%}")
        print(f"中后期专家一致率 (50-75%): {avg_late_mid:.2%}")
        print(f"后期专家一致率 (75-100%): {avg_late:.2%}")
        
        # 分析智能体是否逐渐发展出自己的策略
        if avg_early > avg_late:
            print(f"策略发展分析: 智能体从依赖专家指导({avg_early:.2%})逐渐发展出自己的策略({avg_late:.2%})")
        elif avg_late > avg_early * 1.1:
            print(f"策略发展分析: 智能体随训练进行越来越接近专家策略 ({avg_early:.2%} → {avg_late:.2%})")
        else:
            print(f"策略发展分析: 智能体与专家策略的一致率相对稳定 ({avg_early:.2%} → {avg_late:.2%})")
    
    # 绘制最终训练曲线
    plot_training_curves(
        rewards, losses, win_rates, episode_lengths,
        moving_avg_reward=moving_avg_reward.copy(), 
        moving_avg_loss=moving_avg_loss.copy(), 
        moving_avg_win_rate=moving_avg_win_rate.copy(), 
        episode=episodes,
        expert_agreements=expert_agreements.copy(),
        moving_avg_expert_agreement=moving_avg_expert_agreement.copy(),
        use_expert_guidance=use_expert_guidance
    )
    
    # 如果启用了专家策略学习，额外绘制最终的专家一致率与性能关系图
    if use_expert_guidance and len(expert_agreements) > 0:
        plot_expert_performance_relation(
            rewards, win_rates, expert_agreements,
            episode=episodes
        )
    
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