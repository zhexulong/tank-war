import os
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from ai.base_agent import BaseAgent
from ai.agent import DQNAgent
from ai.logic_agent import LogicAgent, LogicExpertAgent
from ai.environment import TankBattleEnv
from ai.simplified_env import SimplifiedGameEnv
from ai.training import plot_training_curves, plot_expert_performance_relation
from ai.ppo_agent import PPOAgent  # 导入PPO智能体
import torch
import io
import multiprocessing

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

MAX_STEPS_PER_EPISODE_WORKER = 1500 # Max steps per episode in a worker to prevent deadlocks

def worker_collect_episode_data(args_bundle):
    """
    工作者函数，用于收集一个回合的经验数据。
    """
    (worker_id, q_network_state_dict_bytes, state_shape, action_dim,
     current_epsilon, use_expert, expert_reward_init_val, expert_decay_factor_val,
     current_episode_num, total_episodes_num, render_in_worker,
     log_actions_in_worker) = args_bundle

    env_render_mode = 'human' if render_in_worker else None
    env = SimplifiedGameEnv(render_mode=env_render_mode)
    
    # 工作进程始终使用CPU
    device = 'cpu'
    rl_agent_worker = DQNAgent(state_shape, action_dim, device=device)
    if q_network_state_dict_bytes:
        buffer = io.BytesIO(q_network_state_dict_bytes)
        # 确保从GPU加载的状态字典被映射到CPU
        rl_agent_worker.q_network.load_state_dict(
            torch.load(buffer, map_location=device)
        )
    rl_agent_worker.epsilon = current_epsilon # Set worker's epsilon

    expert_agent_worker = LogicExpertAgent()
    logic_agent_opponent_worker = LogicAgent() # Opponent

    experiences = []
    state = env.reset()
    
    episode_reward_val = 0
    episode_steps_val = 0
    expert_agreements_val = 0
    done_val = False
    win_val = False

    while not done_val and episode_steps_val < MAX_STEPS_PER_EPISODE_WORKER:
        rl_action = rl_agent_worker.select_action(state, training=True)
        logic_action_opponent = logic_agent_opponent_worker.select_action(state) # Opponent's action

        # 专家指导逻辑
        step_expert_reward = 0
        matched_expert = False
        if use_expert:
            expert_action = expert_agent_worker.select_action(state)
            # 假设的动作映射和匹配逻辑 (与原train_against_logic中类似)
            # RL: 0-stay, 1-fwd, 2-bwd, 3-left, 4-right, 5-fire
            # Expert: 0-stay, 1-fwd, 2-left, 3-right, 4-fire
            action_map_rl_to_expert = {0:0, 1:1, 3:2, 4:3, 5:4} # RL后退(2)无匹配
            if rl_action in action_map_rl_to_expert and action_map_rl_to_expert[rl_action] == expert_action:
                matched_expert = True
                expert_agreements_val += 1
                current_expert_value = _calculate_expert_reward(current_episode_num, total_episodes_num, expert_reward_init_val, expert_decay_factor_val)
                step_expert_reward = current_expert_value
            
            if log_actions_in_worker and worker_id == 0 : # Log only for worker 0 if enabled
                print(f"  Worker {worker_id} Ep {current_episode_num} Step {episode_steps_val + 1}: RL act: {_action_to_str_dqn(rl_action)}, Exp act: {_action_to_str_logic(expert_action)}, Match: {matched_expert}, Exp Rew: {step_expert_reward:.2f}")


        next_state, step_rewards, done_val, info = env.step([rl_action, logic_action_opponent])
        
        final_reward_for_rl = step_rewards[0] + step_expert_reward # RL is player 1
        
        experiences.append((state, rl_action, final_reward_for_rl, next_state, done_val))
        
        state = next_state
        episode_reward_val += final_reward_for_rl
        episode_steps_val += 1

    if episode_steps_val >= MAX_STEPS_PER_EPISODE_WORKER:
        print(f"Warning: Worker {worker_id} Ep {current_episode_num} reached max steps {MAX_STEPS_PER_EPISODE_WORKER}.")
        # If max steps reached, ensure 'done' is true for the experience
        if experiences:
            s, a, r, ns, _ = experiences[-1]
            experiences[-1] = (s, a, r, ns, True)


    win_val = info.get('winner', 0) == 1 # RL智能体是玩家1
    expert_agreement_rate_val = expert_agreements_val / max(1, episode_steps_val) if use_expert else 0.0
    
    env.close()
    return experiences, episode_reward_val, win_val, episode_steps_val, expert_agreement_rate_val, current_episode_num


def worker_collect_episode_data_ppo(args_bundle):
    """
    工作者函数，用于PPO智能体收集一个回合的经验数据。
    """
    (worker_id, network_state_dict_bytes, state_shape, action_dim,
     use_expert, expert_reward_init_val, expert_decay_factor_val,
     current_episode_num, total_episodes_num, render_in_worker,
     log_actions_in_worker) = args_bundle

    env_render_mode = 'human' if render_in_worker else None
    env = SimplifiedGameEnv(render_mode=env_render_mode)
    
    # 工作进程始终使用CPU
    device = 'cpu'
    rl_agent_worker = PPOAgent(state_shape, action_dim, device=device)
    if network_state_dict_bytes:
        buffer = io.BytesIO(network_state_dict_bytes)
        # 确保从GPU加载的状态字典被映射到CPU
        rl_agent_worker.network.load_state_dict(
            torch.load(buffer, map_location=device)
        )

    expert_agent_worker = LogicExpertAgent()
    logic_agent_opponent_worker = LogicAgent() # 对手

    experiences = []
    state = env.reset()
    
    episode_reward_val = 0
    episode_steps_val = 0
    expert_agreements_val = 0
    done_val = False
    win_val = False

    while not done_val and episode_steps_val < MAX_STEPS_PER_EPISODE_WORKER:
        rl_action = rl_agent_worker.select_action(state, training=True)
        logic_action_opponent = logic_agent_opponent_worker.select_action(state) # 对手的动作

        # 专家指导逻辑
        step_expert_reward = 0
        matched_expert = False
        if use_expert:
            expert_action = expert_agent_worker.select_action(state)
            # 动作映射: PPO到专家
            # PPO: 0-stay, 1-fwd, 2-bwd, 3-left, 4-right, 5-fire
            # Expert: 0-stay, 1-fwd, 2-left, 3-right, 4-fire
            action_map_ppo_to_expert = {0:0, 1:1, 3:2, 4:3, 5:4} # PPO后退(2)无匹配
            if rl_action in action_map_ppo_to_expert and action_map_ppo_to_expert[rl_action] == expert_action:
                matched_expert = True
                expert_agreements_val += 1
                current_expert_value = _calculate_expert_reward(current_episode_num, total_episodes_num, expert_reward_init_val, expert_decay_factor_val)
                step_expert_reward = current_expert_value
            
            if log_actions_in_worker and worker_id == 0: # 仅对worker 0进行日志记录（如果启用）
                print(f"  Worker {worker_id} Ep {current_episode_num} Step {episode_steps_val + 1}: PPO act: {_action_to_str_dqn(rl_action)}, Exp act: {_action_to_str_logic(expert_action)}, Match: {matched_expert}, Exp Rew: {step_expert_reward:.2f}")

        next_state, step_rewards, done_val, info = env.step([rl_action, logic_action_opponent])
        
        final_reward_for_rl = step_rewards[0] + step_expert_reward # RL是玩家1
        
        # 存储PPO所需的信息
        if hasattr(rl_agent_worker, 'last_action_info'):
            log_prob = rl_agent_worker.last_action_info.get('log_prob', 0.0)
            value = rl_agent_worker.last_action_info.get('value', 0.0)
            rl_agent_worker.store_transition(state, rl_action, final_reward_for_rl, next_state, done_val, 
                                            {'log_prob': log_prob, 'value': value})
        else:
            rl_agent_worker.store_transition(state, rl_action, final_reward_for_rl, next_state, done_val)
        
        state = next_state
        episode_reward_val += final_reward_for_rl
        episode_steps_val += 1

    if episode_steps_val >= MAX_STEPS_PER_EPISODE_WORKER:
        print(f"Warning: Worker {worker_id} Ep {current_episode_num} reached max steps {MAX_STEPS_PER_EPISODE_WORKER}.")

    win_val = info.get('winner', 0) == 1 # RL智能体是玩家1
    expert_agreement_rate_val = expert_agreements_val / max(1, episode_steps_val) if use_expert else 0.0
    
    env.close()
    
    # 对于PPO，我们将经验列表与智能体一起返回
    return rl_agent_worker, episode_reward_val, win_val, episode_steps_val, expert_agreement_rate_val, current_episode_num


def train_against_logic(
    episodes: int = 1000, 
    save_interval: int = 100, 
    render: bool = False, 
    checkpoint_path: str = None,
    use_expert_guidance: bool = False,
    expert_reward_init: float = 100.0,
    expert_decay_factor: float = 1.0,
    num_workers: int = 1, # 新增并行工作者数量参数
    log_worker_actions: bool = False # 新增: 是否打印首个worker的详细动作日志
):
    """训练RL智能体对抗Logic智能体
    
    Args:
        episodes: 训练回合数
        save_interval: 模型保存间隔
        render: 是否渲染游戏画面
        checkpoint_path: 检查点路径，如果存在则从检查点继续训练
        use_expert_guidance: 是否使用专家策略学习
        expert_reward_init: 初始专家奖励值
        expert_decay_factor: 专家奖励衰减因子 (1.0表示完全按照训练进度线性衰减，0表示不衰减)
    """
    # 生成训练运行的时间戳（精确到分钟）
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # 创建模型保存目录
    model_dir = os.path.join('models', run_timestamp)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n训练开始时间: {run_timestamp}")
    print(f"模型将保存在: {model_dir}")
    print(f"训练曲线将保存在: output/{run_timestamp}\n")
    
    # 创建环境 (主进程的环境仅用于获取参数或单进程模式)
    if num_workers <= 1: # Or if render is True and num_workers > 1, main env might be needed for rendering
        env = SimplifiedGameEnv(render_mode='human' if render else None)
    else: # In parallel mode, create a temporary env to get params
        temp_env_for_params = SimplifiedGameEnv(render_mode=None)
        env = temp_env_for_params # Assign to env for later parameter extraction, will be closed.

    state_shape = {
        'map': env.observation_space['map'],
        'tanks': env.observation_space['tanks'],
        'bullets': env.observation_space['bullets']
    }
    action_dim = env.action_space.n
    
    if num_workers > 1 and not (render and num_workers > 1) : # Close temp env if it was created
         if 'temp_env_for_params' in locals() and temp_env_for_params is env :
              env.close() # Close the temporary env
              env = None # Set env to None as workers will handle their own

    # 主进程的DQNAgent在GPU上创建（如果可用）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rl_agent = DQNAgent(state_shape, action_dim, device=device)
    # logic_agent and expert_agent are instantiated in workers if num_workers > 1
    if num_workers <= 1:
        logic_agent = LogicAgent()
        expert_agent = LogicExpertAgent()
    
    # Print device information
    print(f"Main process using device: {device}")
    
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
    if num_workers <= 1:
        # Original single-process training loop
        for episode in range(start_episode, episodes + 1):
            # 重置环境
            state = env.reset()
            episode_reward = 0
            episode_loss_sum = 0 # Sum of losses in the episode
            episode_step = 0
            expert_agreement_count = 0
            done = False
            
            current_expert_reward_value = _calculate_expert_reward(episode, episodes, expert_reward_init, expert_decay_factor)

            while not done and episode_step < MAX_STEPS_PER_EPISODE_WORKER : # Added max step protection
                # RL智能体动作
                rl_action = rl_agent.select_action(state, training=True)
                
                # 对手智能体动作
                logic_action = logic_agent.select_action(state)
                
                # 专家指导
                step_expert_reward = 0
                matched_expert = False
                if use_expert_guidance:
                    expert_action = expert_agent.select_action(state)
                    # 动作映射: RL到专家 (0:0, 1:1, 3:2, 4:3, 5:4)
                    action_map_rl_to_expert = {0:0, 1:1, 3:2, 4:3, 5:4}
                    if rl_action in action_map_rl_to_expert and action_map_rl_to_expert[rl_action] == expert_action:
                        matched_expert = True
                        expert_agreement_count += 1
                        step_expert_reward = current_expert_reward_value # 使用当前回合计算的专家奖励值
                    
                    if log_worker_actions: # For single worker, this acts as main log
                         print(f"  Main Ep {episode} Step {episode_step + 1}: RL act: {_action_to_str_dqn(rl_action)}, Exp act: {_action_to_str_logic(expert_action)}, Match: {matched_expert}, Exp Rew: {step_expert_reward:.2f}")


                # 执行动作
                next_state, step_rewards, done, info = env.step([rl_action, logic_action])
                
                # 计算最终奖励 (RL智能体是玩家1)
                final_reward = step_rewards[0] + step_expert_reward
                
                # 存储经验
                rl_agent.replay_buffer.push(state, rl_action, final_reward, next_state, done)
                
                # 更新状态和奖励
                state = next_state
                episode_reward += final_reward
                episode_step += 1
                
                # 训练RL智能体
                loss = rl_agent.train()
                if loss is not None:
                    episode_loss_sum += loss
            
            if episode_step >= MAX_STEPS_PER_EPISODE_WORKER:
                print(f"Warning: Main Ep {episode} reached max steps {MAX_STEPS_PER_EPISODE_WORKER}.")


            # 记录训练数据
            rewards.append(episode_reward)
            losses.append(episode_loss_sum / max(1, episode_step)) # Average loss for the episode
            win = info.get('winner', 0) == 1
            win_rates.append(1 if win else 0)
            episode_lengths.append(episode_step)
            
            expert_agreement_rate = expert_agreement_count / max(1, episode_step) if use_expert_guidance else 0.0
            expert_agreements.append(expert_agreement_rate)
            
            # 打印训练信息
            expert_info_str = ""
            if use_expert_guidance:
                expert_info_str = f", ExpAgree: {expert_agreement_rate:.2%}, ExpRewVal: {current_expert_reward_value:.2f}"
            
            print(f"Episode {episode}/{episodes}, Reward: {episode_reward:.2f}, AvgLoss: {losses[-1]:.4f}, "
                  f"Win: {win}, Steps: {episode_step}, Epsilon: {rl_agent.epsilon:.4f}{expert_info_str}")

            # 计算移动平均
            current_window_size = min(len(rewards), window_size)
            if current_window_size > 0:
                moving_avg_reward.append(np.mean(rewards[-current_window_size:]))
                moving_avg_loss.append(np.mean(losses[-current_window_size:]))
                moving_avg_win_rate.append(np.mean(win_rates[-current_window_size:]))
                if use_expert_guidance:
                    moving_avg_expert_agreement.append(np.mean(expert_agreements[-current_window_size:]))
            
            if episode % save_interval == 0:
                checkpoint_data = {
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
                    'epsilon': rl_agent.epsilon,
                    'use_expert_guidance': use_expert_guidance,
                    'expert_reward_init': expert_reward_init,
                    'expert_decay_factor': expert_decay_factor
                }
                chkpt_path = os.path.join(model_dir, f"rl_vs_logic_checkpoint_{episode}.pt")
                torch.save(checkpoint_data, chkpt_path)
                print(f"已保存检查点到: {chkpt_path}")

                # 定期绘制训练曲线图
                plot_training_curves(
                    rewards, losses, win_rates, episode_lengths,
                    moving_avg_reward, moving_avg_loss, moving_avg_win_rate,
                    episode, expert_agreements, moving_avg_expert_agreement,
                    use_expert_guidance, run_timestamp
                )
                if use_expert_guidance:
                    plot_expert_performance_relation(
                        rewards, win_rates, expert_agreements, episode, run_timestamp=run_timestamp
                    )
    else: # Parallel training loop using multiprocessing
        # Ensure 'spawn' start method for better compatibility, especially on Windows with Pygame
        # However, 'fork' (default on Linux) is generally more efficient if it works.
        # For simplicity here, we'll use the default, but consider get_context('spawn') if issues arise.
        import multiprocessing as mp # Import at a more local scope or ensure it's at the top
        ctx = mp.get_context('spawn') # Use spawn context for CUDA compatibility
        
        processed_episodes_count = start_episode - 1 # Initialize before the loop

        with ctx.Pool(processes=num_workers) as pool: # Use the spawn context pool
            while processed_episodes_count < episodes:
                num_episodes_to_run_in_batch = min(num_workers, episodes - processed_episodes_count)
                if num_episodes_to_run_in_batch <= 0:
                    break

                q_network_state_dict_bytes = None
                if rl_agent.q_network:
                    buffer = io.BytesIO()
                    torch.save(rl_agent.q_network.state_dict(), buffer)
                    q_network_state_dict_bytes = buffer.getvalue()

                tasks_args = []
                for i in range(num_episodes_to_run_in_batch):
                    current_global_episode_num = processed_episodes_count + 1 + i
                    render_for_this_worker = render and i == 0 

                    tasks_args.append((
                        i, q_network_state_dict_bytes, state_shape, action_dim,
                        rl_agent.epsilon, use_expert_guidance, expert_reward_init, expert_decay_factor,
                        current_global_episode_num, episodes, render_for_this_worker,
                        log_worker_actions
                    ))
                
                batch_results = pool.map(worker_collect_episode_data, tasks_args)
                

                for experiences_list, ep_reward, ep_win, ep_steps, ep_expert_agree_rate, actual_ep_num_returned in batch_results:
                    processed_episodes_count = actual_ep_num_returned 

                    for exp in experiences_list:
                        rl_agent.replay_buffer.push(*exp)
                    
                    loss_item = rl_agent.train() 
                    
                    # Append per-episode metrics
                    rewards.append(ep_reward)
                    win_rates.append(1 if ep_win else 0)
                    episode_lengths.append(ep_steps)
                    
                    if use_expert_guidance:
                        expert_agreements.append(ep_expert_agree_rate)
                    else:
                        expert_agreements.append(0.0) # Keep length consistent

                    if loss_item is not None:
                        losses.append(loss_item)
                    elif losses: 
                        losses.append(losses[-1]) # Append previous loss if current is None
                    else:
                        losses.append(0) # Append 0 if losses list is empty and current loss is None

                    # Calculate and append moving averages for THIS episode
                    current_idx = len(rewards) - 1 
                    window_start_idx = max(0, current_idx - window_size + 1)

                    moving_avg_reward.append(np.mean(rewards[window_start_idx : current_idx + 1]))
                    moving_avg_loss.append(np.mean(losses[window_start_idx : current_idx + 1]))
                    moving_avg_win_rate.append(np.mean(win_rates[window_start_idx : current_idx + 1]))
                    
                    if use_expert_guidance:
                        moving_avg_expert_agreement.append(np.mean(expert_agreements[window_start_idx : current_idx + 1]))
                    elif moving_avg_expert_agreement: 
                        moving_avg_expert_agreement.append(moving_avg_expert_agreement[-1]) 
                    else: 
                        moving_avg_expert_agreement.append(0)

                    current_expert_reward_value_parallel = _calculate_expert_reward(actual_ep_num_returned, episodes, expert_reward_init, expert_decay_factor)
                    expert_info_str_parallel = ""
                    if use_expert_guidance:
                         expert_info_str_parallel = f", ExpAgree: {ep_expert_agree_rate:.2%}, ExpRewVal: {current_expert_reward_value_parallel:.2f}"

                    print(f"Worker (Ep {actual_ep_num_returned}) finished. Reward: {ep_reward:.2f}, "
                          f"Win: {ep_win}, Steps: {ep_steps}, Epsilon: {rl_agent.epsilon:.4f}{expert_info_str_parallel}")

                # 完全重写的保存逻辑：检查每个已完成的episode，是否为save_interval的倍数
                # 计算这一批次中需要保存的所有episode点
                for check_ep in range(processed_episodes_count - len(batch_results) + 1, processed_episodes_count + 1):
                    if check_ep % save_interval == 0 or check_ep == episodes:
                        # 这个检查点需要保存
                        checkpoint_data = {
                            'episode': check_ep,
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
                            'epsilon': rl_agent.epsilon,
                            'use_expert_guidance': use_expert_guidance,
                            'expert_reward_init': expert_reward_init,
                            'expert_decay_factor': expert_decay_factor
                        }
                        
                        # 文件名使用当前episode数
                        chkpt_path = os.path.join(model_dir, f"rl_vs_logic_checkpoint_{check_ep}.pt")
                        torch.save(checkpoint_data, chkpt_path)
                        print(f"已保存检查点到: {chkpt_path}")

                        # 绘制训练曲线
                        plot_training_curves(
                            rewards, losses, win_rates, episode_lengths,
                            moving_avg_reward, moving_avg_loss, moving_avg_win_rate,
                            check_ep, expert_agreements, moving_avg_expert_agreement,
                            use_expert_guidance, run_timestamp
                        )
                        
                        if use_expert_guidance:
                            plot_expert_performance_relation(
                                rewards, win_rates, expert_agreements, check_ep, run_timestamp=run_timestamp
                            )
                if processed_episodes_count >= episodes:
                    break


    # 保存最终模型到时间戳目录
    final_model_path = os.path.join(model_dir, "rl_vs_logic_final.pt")
    rl_agent.save(final_model_path)
    print(f"最终模型已保存到: {final_model_path}")
    
    # 关闭主环境（如果已创建且未使用temp env logic）
    if env: # env might be None if num_workers > 1 and not render
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
    final_episode_count_for_plot = episodes if num_workers <=1 else processed_episodes_count
    plot_training_curves(
        rewards, losses, win_rates, episode_lengths,
        moving_avg_reward=moving_avg_reward, # Pass the list directly
        moving_avg_loss=moving_avg_loss, 
        moving_avg_win_rate=moving_avg_win_rate, 
        episode=final_episode_count_for_plot, # Use actual number of processed episodes
        expert_agreements=expert_agreements,
        moving_avg_expert_agreement=moving_avg_expert_agreement,
        use_expert_guidance=use_expert_guidance,
        run_timestamp=run_timestamp
    )
    
    # 如果启用了专家策略学习，额外绘制最终的专家一致率与性能关系图
    if use_expert_guidance and len(expert_agreements) > 0:
        plot_expert_performance_relation(
            rewards, win_rates, expert_agreements, final_episode_count_for_plot, run_timestamp=run_timestamp
        )
    
    return rl_agent

def evaluate_agents(rl_agent: BaseAgent, logic_agent: BaseAgent, num_episodes: int = 100, render: bool = False) -> Tuple[float, float]:
    """评估两个智能体的对抗性能
    
    Args:
        rl_agent: 强化学习智能体 (DQN或PPO)
        logic_agent: Logic智能体
        num_episodes: 评估局数
        render: 是否渲染游戏画面
        
    Returns:
        win_rate: RL智能体的胜率
        avg_reward: RL智能体的平均奖励
    """
    env = SimplifiedGameEnv(render_mode='human' if render else None)
    
    total_wins = 0
    total_reward = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 选择动作
            rl_action = rl_agent.select_action(state, training=False)
            logic_action = logic_agent.select_action(state)
            
            # 执行动作
            next_state, rewards, done, info = env.step([rl_action, logic_action])
            
            # 更新奖励
            episode_reward += rewards[0]
            
            # 更新状态
            state = next_state
            
            # 渲染
            if render:
                time.sleep(0.01)
        
        # 记录胜利
        if info.get('winner', 0) == 1:  # RL智能体是玩家1
            total_wins += 1
        
        total_reward += episode_reward
        
        # 打印评估信息
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Win: {info.get('winner', 0) == 1}")
    
    env.close()
    
    win_rate = total_wins / num_episodes
    avg_reward = total_reward / num_episodes
    
    print(f"\n评估结果:")
    print(f"胜率: {win_rate:.2%}")
    print(f"平均奖励: {avg_reward:.2f}")
    
    return win_rate, avg_reward

def train_ppo_against_logic(
    episodes: int = 1000, 
    save_interval: int = 100, 
    render: bool = False, 
    checkpoint_path: str = None,
    use_expert_guidance: bool = False,
    expert_reward_init: float = 100.0,
    expert_decay_factor: float = 1.0,
    num_workers: int = 1, 
    log_worker_actions: bool = False,
    ppo_epochs: int = 4,         # PPO特有参数：每批数据训练轮数
    ppo_batch_size: int = 64,    # PPO特有参数：批次大小
):
    """使用PPO智能体对抗Logic智能体进行训练
    
    Args:
        episodes: 训练回合数
        save_interval: 模型保存间隔
        render: 是否渲染游戏画面
        checkpoint_path: 检查点路径，如果存在则从检查点继续训练
        use_expert_guidance: 是否使用专家策略学习
        expert_reward_init: 初始专家奖励值
        expert_decay_factor: 专家奖励衰减因子 (1.0表示完全按照训练进度线性衰减)
        num_workers: 并行工作者数量
        log_worker_actions: 是否打印首个工作者的详细动作日志
        ppo_epochs: PPO每批数据训练轮数
        ppo_batch_size: PPO训练批次大小
    """
    # 生成训练运行的时间戳（精确到分钟）
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # 创建模型保存目录
    model_dir = os.path.join('models', run_timestamp)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n训练开始时间: {run_timestamp}")
    print(f"模型将保存在: {model_dir}")
    print(f"训练曲线将保存在: output/{run_timestamp}\n")
    
    # 创建环境 (主进程的环境仅用于获取参数或单进程模式)
    if num_workers <= 1:
        env = SimplifiedGameEnv(render_mode='human' if render else None)
    else: 
        temp_env_for_params = SimplifiedGameEnv(render_mode=None)
        env = temp_env_for_params

    state_shape = {
        'map': env.observation_space['map'],
        'tanks': env.observation_space['tanks'],
        'bullets': env.observation_space['bullets']
    }
    action_dim = env.action_space.n
    
    if num_workers > 1 and not (render and num_workers > 1):
        if 'temp_env_for_params' in locals() and temp_env_for_params is env:
            env.close() 
            env = None

    # 主进程的PPO智能体在GPU上创建（如果可用）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ppo_agent = PPOAgent(state_shape, action_dim, device=device)
    
    if num_workers <= 1:
        logic_agent = LogicAgent()
        expert_agent = LogicExpertAgent()
    
    print(f"主进程使用设备: {device}")
    
    # 创建保存目录
    os.makedirs('models', exist_ok=True)
    
    # 训练记录
    rewards = []
    losses = []
    win_rates = []
    episode_lengths = []
    expert_agreements = []
    
    # 移动平均记录
    window_size = 100
    moving_avg_reward = []
    moving_avg_loss = []
    moving_avg_win_rate = []
    moving_avg_expert_agreement = []
    
    # 加载检查点（如果存在）
    start_episode = 1
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        ppo_agent.network.load_state_dict(checkpoint['model_state_dict'])
        ppo_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        rewards = checkpoint['rewards']
        losses = checkpoint['losses']
        win_rates = checkpoint['win_rates']
        episode_lengths = checkpoint['episode_lengths']
        expert_agreements = checkpoint.get('expert_agreements', [])
        moving_avg_reward = checkpoint.get('moving_avg_reward', [])
        moving_avg_loss = checkpoint.get('moving_avg_loss', [])
        moving_avg_win_rate = checkpoint.get('moving_avg_win_rate', [])
        moving_avg_expert_agreement = checkpoint.get('moving_avg_expert_agreement', [])
            
        start_episode = checkpoint['episode'] + 1
        print(f"加载检查点：{checkpoint_path}，从第{start_episode}轮继续训练")
    
    # 训练循环
    if num_workers <= 1:
        # 单进程训练循环
        for episode in range(start_episode, episodes + 1):
            state = env.reset()
            episode_reward = 0
            episode_step = 0
            expert_agreement_count = 0
            done = False
            
            current_expert_reward_value = _calculate_expert_reward(episode, episodes, expert_reward_init, expert_decay_factor)

            while not done and episode_step < MAX_STEPS_PER_EPISODE_WORKER:
                # PPO智能体选择动作
                ppo_action = ppo_agent.select_action(state, training=True)
                
                # 对手选择动作
                logic_action = logic_agent.select_action(state)
                
                # 专家指导
                step_expert_reward = 0
                matched_expert = False
                if use_expert_guidance:
                    expert_action = expert_agent.select_action(state)
                    # 动作映射: PPO到专家
                    action_map_ppo_to_expert = {0:0, 1:1, 3:2, 4:3, 5:4}
                    if ppo_action in action_map_ppo_to_expert and action_map_ppo_to_expert[ppo_action] == expert_action:
                        matched_expert = True
                        expert_agreement_count += 1
                        step_expert_reward = current_expert_reward_value
                    
                    if log_worker_actions:
                        print(f"  Main Ep {episode} Step {episode_step + 1}: PPO act: {_action_to_str_dqn(ppo_action)}, Exp act: {_action_to_str_logic(expert_action)}, Match: {matched_expert}, Exp Rew: {step_expert_reward:.2f}")

                # 执行动作
                next_state, step_rewards, done, info = env.step([ppo_action, logic_action])
                
                # 计算最终奖励 (PPO智能体是玩家1)
                final_reward = step_rewards[0] + step_expert_reward
                
                # 存储经验
                if hasattr(ppo_agent, 'last_action_info'):
                    log_prob = ppo_agent.last_action_info.get('log_prob', 0.0)
                    value = ppo_agent.last_action_info.get('value', 0.0)
                    ppo_agent.store_transition(state, ppo_action, final_reward, next_state, done, 
                                             {'log_prob': log_prob, 'value': value})
                else:
                    ppo_agent.store_transition(state, ppo_action, final_reward, next_state, done)
                
                state = next_state
                episode_reward += final_reward
                episode_step += 1
            
            if episode_step >= MAX_STEPS_PER_EPISODE_WORKER:
                print(f"Warning: Main Ep {episode} reached max steps {MAX_STEPS_PER_EPISODE_WORKER}.")

            # 训练PPO智能体
            loss = ppo_agent.train(batch_size=ppo_batch_size, epochs=ppo_epochs)
            
            # 记录训练数据
            rewards.append(episode_reward)
            losses.append(loss if loss is not None else 0.0)
            win = info.get('winner', 0) == 1
            win_rates.append(1 if win else 0)
            episode_lengths.append(episode_step)
            
            expert_agreement_rate = expert_agreement_count / max(1, episode_step) if use_expert_guidance else 0.0
            expert_agreements.append(expert_agreement_rate)
            
            # 打印训练信息
            expert_info_str = ""
            if use_expert_guidance:
                expert_info_str = f", ExpAgree: {expert_agreement_rate:.2%}, ExpRewVal: {current_expert_reward_value:.2f}"
            
            print(f"Episode {episode}/{episodes}, Reward: {episode_reward:.2f}, AvgLoss: {losses[-1]:.4f}, "
                  f"Win: {win}, Steps: {episode_step}, Epsilon: {ppo_agent.epsilon:.4f}{expert_info_str}")

            # 计算移动平均
            current_window_size = min(len(rewards), window_size)
            if current_window_size > 0:
                moving_avg_reward.append(np.mean(rewards[-current_window_size:]))
                moving_avg_loss.append(np.mean(losses[-current_window_size:]))
                moving_avg_win_rate.append(np.mean(win_rates[-current_window_size:]))
                if use_expert_guidance:
                    moving_avg_expert_agreement.append(np.mean(expert_agreements[-current_window_size:]))
            
            if episode % save_interval == 0:
                checkpoint_data = {
                    'episode': episode,
                    'model_state_dict': ppo_agent.network.state_dict(),
                    'optimizer_state_dict': ppo_agent.optimizer.state_dict(),
                    'rewards': rewards,
                    'losses': losses,
                    'win_rates': win_rates,
                    'episode_lengths': episode_lengths,
                    'expert_agreements': expert_agreements,
                    'moving_avg_reward': moving_avg_reward,
                    'moving_avg_loss': moving_avg_loss,
                    'moving_avg_win_rate': moving_avg_win_rate,
                    'moving_avg_expert_agreement': moving_avg_expert_agreement,
                    'use_expert_guidance': use_expert_guidance,
                    'expert_reward_init': expert_reward_init,
                    'expert_decay_factor': expert_decay_factor
                }
                chkpt_path = os.path.join(model_dir, f"ppo_vs_logic_checkpoint_{episode}.pt")
                torch.save(checkpoint_data, chkpt_path)
                print(f"已保存检查点到: {chkpt_path}")

                # 绘制训练曲线图
                plot_training_curves(
                    rewards, losses, win_rates, episode_lengths,
                    moving_avg_reward, moving_avg_loss, moving_avg_win_rate,
                    episode, expert_agreements, moving_avg_expert_agreement,
                    use_expert_guidance, run_timestamp
                )
                if use_expert_guidance:
                    plot_expert_performance_relation(
                        rewards, win_rates, expert_agreements, episode, run_timestamp=run_timestamp
                    )
    else:
        # 并行训练循环
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        
        processed_episodes_count = start_episode - 1

        with ctx.Pool(processes=num_workers) as pool:
            while processed_episodes_count < episodes:
                num_episodes_to_run_in_batch = min(num_workers, episodes - processed_episodes_count)
                if num_episodes_to_run_in_batch <= 0:
                    break

                network_state_dict_bytes = None
                if ppo_agent.network:
                    buffer = io.BytesIO()
                    torch.save(ppo_agent.network.state_dict(), buffer)
                    network_state_dict_bytes = buffer.getvalue()

                tasks_args = []
                for i in range(num_episodes_to_run_in_batch):
                    current_global_episode_num = processed_episodes_count + 1 + i
                    render_for_this_worker = render and i == 0 

                    tasks_args.append((
                        i, network_state_dict_bytes, state_shape, action_dim,
                        use_expert_guidance, expert_reward_init, expert_decay_factor,
                        current_global_episode_num, episodes, render_for_this_worker,
                        log_worker_actions
                    ))
                
                batch_results = pool.map(worker_collect_episode_data_ppo, tasks_args)
                
                for worker_agent, ep_reward, ep_win, ep_steps, ep_expert_agree_rate, actual_ep_num_returned in batch_results:
                    processed_episodes_count = actual_ep_num_returned
                    
                    # 主PPO智能体从工作者PPO智能体中合并经验
                    if hasattr(worker_agent, 'buffer'):
                        for exp_data in zip(worker_agent.buffer.states, 
                                           worker_agent.buffer.actions,
                                           worker_agent.buffer.rewards,
                                           worker_agent.buffer.next_states,
                                           worker_agent.buffer.dones,
                                           zip(worker_agent.buffer.log_probs, worker_agent.buffer.values)):
                            s, a, r, ns, d, (lp, v) = exp_data
                            ppo_agent.store_transition(s, a, r, ns, d, {'log_prob': lp, 'value': v})
                    
                    # 训练主PPO智能体
                    loss_item = ppo_agent.train(batch_size=ppo_batch_size, epochs=ppo_epochs)
                    
                    # 记录训练数据
                    rewards.append(ep_reward)
                    win_rates.append(1 if ep_win else 0)
                    episode_lengths.append(ep_steps)
                    
                    if use_expert_guidance:
                        expert_agreements.append(ep_expert_agree_rate)
                    else:
                        expert_agreements.append(0.0)

                    if loss_item is not None:
                        losses.append(loss_item)
                    elif losses: 
                        losses.append(losses[-1])
                    else:
                        losses.append(0)

                    # 计算并记录移动平均
                    current_idx = len(rewards) - 1 
                    window_start_idx = max(0, current_idx - window_size + 1)

                    moving_avg_reward.append(np.mean(rewards[window_start_idx : current_idx + 1]))
                    moving_avg_loss.append(np.mean(losses[window_start_idx : current_idx + 1]))
                    moving_avg_win_rate.append(np.mean(win_rates[window_start_idx : current_idx + 1]))
                    
                    if use_expert_guidance:
                        moving_avg_expert_agreement.append(np.mean(expert_agreements[window_start_idx : current_idx + 1]))
                    elif moving_avg_expert_agreement: 
                        moving_avg_expert_agreement.append(moving_avg_expert_agreement[-1]) 
                    else: 
                        moving_avg_expert_agreement.append(0)

                    current_expert_reward_value_parallel = _calculate_expert_reward(actual_ep_num_returned, episodes, expert_reward_init, expert_decay_factor)
                    expert_info_str_parallel = ""
                    if use_expert_guidance:
                         expert_info_str_parallel = f", ExpAgree: {ep_expert_agree_rate:.2%}, ExpRewVal: {current_expert_reward_value_parallel:.2f}"

                    print(f"Worker (Ep {actual_ep_num_returned}) finished. Reward: {ep_reward:.2f}, "
                          f"Win: {ep_win}, Steps: {ep_steps}{expert_info_str_parallel}")

                # 检查是否需要保存检查点
                for check_ep in range(processed_episodes_count - len(batch_results) + 1, processed_episodes_count + 1):
                    if check_ep % save_interval == 0 or check_ep == episodes:
                        checkpoint_data = {
                            'episode': check_ep,
                            'model_state_dict': ppo_agent.network.state_dict(),
                            'optimizer_state_dict': ppo_agent.optimizer.state_dict(),
                            'rewards': rewards,
                            'losses': losses,
                            'win_rates': win_rates,
                            'episode_lengths': episode_lengths,
                            'expert_agreements': expert_agreements,
                            'moving_avg_reward': moving_avg_reward,
                            'moving_avg_loss': moving_avg_loss,
                            'moving_avg_win_rate': moving_avg_win_rate,
                            'moving_avg_expert_agreement': moving_avg_expert_agreement,
                            'use_expert_guidance': use_expert_guidance,
                            'expert_reward_init': expert_reward_init,
                            'expert_decay_factor': expert_decay_factor
                        }
                        
                        chkpt_path = os.path.join(model_dir, f"ppo_vs_logic_checkpoint_{check_ep}.pt")
                        torch.save(checkpoint_data, chkpt_path)
                        print(f"已保存检查点到: {chkpt_path} (episode {check_ep})")
                        
                        # 绘制训练曲线
                        plot_training_curves(
                            rewards, losses, win_rates, episode_lengths,
                            moving_avg_reward, moving_avg_loss, moving_avg_win_rate,
                            check_ep, expert_agreements, moving_avg_expert_agreement,
                            use_expert_guidance, run_timestamp
                        )
                        
                        if use_expert_guidance:
                            plot_expert_performance_relation(
                                rewards, win_rates, expert_agreements, check_ep, run_timestamp=run_timestamp
                            )
                if processed_episodes_count >= episodes:
                    break

    # 保存最终模型
    final_model_path = os.path.join(model_dir, "ppo_vs_logic_final.pt")
    ppo_agent.save_checkpoint(final_model_path)
    print(f"最终模型已保存到: {final_model_path}")
    
    # 关闭环境
    if env:
        env.close()
    
    # 输出最终训练结果摘要
    if len(moving_avg_reward) > 0:
        print(f"\n===== 训练结果摘要 =====")
        print(f"最终移动平均奖励: {moving_avg_reward[-1]:.4f}")
        print(f"最终移动平均损失: {moving_avg_loss[-1]:.4f}")
        print(f"最终移动平均胜率: {moving_avg_win_rate[-1]:.2%}")
    
    # 专家策略学习分析
    if use_expert_guidance and len(expert_agreements) > 0:
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
        
        if avg_early > avg_late:
            print(f"策略发展分析: 智能体从依赖专家指导({avg_early:.2%})逐渐发展出自己的策略({avg_late:.2%})")
        elif avg_late > avg_early * 1.1:
            print(f"策略发展分析: 智能体随训练进行越来越接近专家策略 ({avg_early:.2%} → {avg_late:.2%})")
        else:
            print(f"策略发展分析: 智能体与专家策略的一致率相对稳定 ({avg_early:.2%} → {avg_late:.2%})")
    
    # 绘制最终训练曲线
    final_episode_count_for_plot = episodes if num_workers <=1 else processed_episodes_count
    plot_training_curves(
        rewards, losses, win_rates, episode_lengths,
        moving_avg_reward, moving_avg_loss, moving_avg_win_rate,
        final_episode_count_for_plot,
        expert_agreements, moving_avg_expert_agreement,
        use_expert_guidance, run_timestamp
    )
    
    if use_expert_guidance and len(expert_agreements) > 0:
        plot_expert_performance_relation(
            rewards, win_rates, expert_agreements, final_episode_count_for_plot, run_timestamp=run_timestamp
        )
    
    return ppo_agent