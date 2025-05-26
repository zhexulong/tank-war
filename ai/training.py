import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def plot_training_curves(rewards, losses, win_rates, episode_lengths, 
                         moving_avg_reward=None, moving_avg_loss=None, moving_avg_win_rate=None, 
                         episode=None, expert_agreements=None, moving_avg_expert_agreement=None,
                         use_expert_guidance=False,
                         run_timestamp=None):
    """绘制训练曲线
    
    Args:
        rewards: 每个回合的奖励列表
        losses: 每个回合的损失列表
        win_rates: 每个回合的胜率列表
        episode_lengths: 每个回合的步数列表
        moving_avg_reward: 奖励移动平均列表
        moving_avg_loss: 损失移动平均列表
        moving_avg_win_rate: 胜率移动平均列表
        episode: 当前回合数
        expert_agreements: 与专家动作一致率列表
        moving_avg_expert_agreement: 与专家动作一致率移动平均列表
        use_expert_guidance: 是否显示专家策略学习指标
        run_timestamp: 训练开始的时间戳字符串
    """
    try:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置输出目录
        if run_timestamp:
            output_dir = os.path.join('output', run_timestamp)
        else:
            output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建图表
        rows = 3 if use_expert_guidance and expert_agreements is not None else 2
        fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))
        
        # 设置axes为合适的格式
        if rows == 2:
            # 2x2 布局
            ax1, ax2 = axes[0]
            ax3, ax4 = axes[1]
            expert_plot = False
        else:
            # 3x2 布局
            ax1, ax2 = axes[0]
            ax3, ax4 = axes[1]
            ax5, ax6 = axes[2]  # 专家策略指标用ax5
            expert_plot = True
        
        # 生成x轴数据
        episodes = range(1, len(rewards) + 1)
        moving_avg_episodes = range(1, len(moving_avg_loss) + 1) if moving_avg_loss is not None else []
        
        # Plot reward curve
        ax1.plot(episodes, rewards, label='Reward', color='gray', alpha=0.3)
        if moving_avg_reward is not None and len(moving_avg_reward) > 0:
            ax1.plot(moving_avg_episodes, moving_avg_reward, label='Moving Average Reward', color='red')
        ax1.set_title('Training Reward Curve')
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Reward Value')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss curve
        loss_episodes = range(1, len(losses) + 1)
        ax2.plot(loss_episodes, losses, label='Loss', color='gray', alpha=0.3)
        if moving_avg_loss is not None and len(moving_avg_loss) > 0:
            ax2.plot(moving_avg_episodes, moving_avg_loss, label='Moving Average Loss', color='blue')
        ax2.set_title('Training Loss Curve')
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Loss Value')
        ax2.legend()
        ax2.grid(True)
        
        # Plot win rate curve
        win_episodes = range(1, len(win_rates) + 1)
        ax3.plot(win_episodes, win_rates, label='Win Rate', color='gray', alpha=0.3)
        if moving_avg_win_rate is not None and len(moving_avg_win_rate) > 0:
            ax3.plot(moving_avg_episodes, moving_avg_win_rate, label='Moving Average Win Rate', color='green')
        ax3.set_title('Win Rate Curve')
        ax3.set_xlabel('Episodes')
        ax3.set_ylabel('Win Rate')
        ax3.legend()
        ax3.grid(True)
        
        # 绘制游戏步数曲线
        episodes = range(1, len(episode_lengths) + 1)
        ax4.plot(episodes, episode_lengths, label='Episode Length', color='gray', alpha=0.3)
        ax4.set_title('Episode Length Curve')
        ax4.set_xlabel('Episodes')
        ax4.set_ylabel('Episode Length')
        ax4.legend()
        ax4.grid(True)
        
        # 如果启用了专家策略学习，绘制额外的指标
        if expert_plot and expert_agreements is not None:
            # 绘制专家一致性曲线
            expert_episodes = range(1, len(expert_agreements) + 1)
            ax5.plot(expert_episodes, expert_agreements, label='Expert Agreement', color='gray', alpha=0.3)
            if moving_avg_expert_agreement is not None and len(moving_avg_expert_agreement) > 0:
                ax5.plot(moving_avg_episodes, moving_avg_expert_agreement, label='Moving Avg Expert Agreement', color='purple')
            ax5.set_title('Expert Policy Agreement Rate')
            ax5.set_xlabel('Episodes')
            ax5.set_ylabel('Agreement Rate')
            ax5.set_ylim(0, 1.0)  # 设置y轴范围为0-1
            ax5.legend()
            ax5.grid(True)
            
            # 空出右下角，或者添加其他感兴趣的指标
            ax6.set_visible(False)  # 暂时不使用第六个图表位置
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if episode is not None:
            # 如果使用了专家策略，在文件名中标注
            expert_suffix = '_with_expert' if use_expert_guidance else ''
            plt.savefig(os.path.join(output_dir, f'training_curves_episode_{episode}{expert_suffix}.png'))
        else:
            expert_suffix = '_with_expert' if use_expert_guidance else ''
            plt.savefig(os.path.join(output_dir, f'training_curves_final{expert_suffix}.png'))
    finally:
        # 确保无论如何都关闭图表
        plt.close('all')

def plot_expert_performance_relation(rewards, win_rates, expert_agreements, episode=None, window_size=100, run_timestamp=None):
    """Plot the relationship between expert agreement rate and performance metrics
    
    Args:
        rewards: List of rewards for each episode
        win_rates: List of win rates for each episode
        expert_agreements: List of expert action agreement rates for each episode
        episode: Current episode number
        window_size: Window size for moving average calculation
        run_timestamp: Training run start timestamp string
    """
    # Only used when expert policy learning is enabled
    if expert_agreements is None or len(expert_agreements) == 0:
        return
    
    try:
        # Calculate moving averages
        moving_avg_rewards = []
        moving_avg_win_rates = []
        moving_avg_agreements = []
        
        # Generate x-axis data
        episodes = range(1, len(expert_agreements) + 1)
        
        # Calculate moving averages
        for i in range(len(expert_agreements)):
            window_start = max(0, i - window_size + 1)
            moving_avg_agreements.append(np.mean(expert_agreements[window_start:i+1]))
            moving_avg_rewards.append(np.mean(rewards[window_start:i+1]))
            moving_avg_win_rates.append(np.mean(win_rates[window_start:i+1]))
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot expert agreement rate over time
        plt.subplot(2, 1, 1)
        plt.plot(episodes, expert_agreements, color='gray', alpha=0.3, label='Per Episode Agreement Rate')
        plt.plot(episodes, moving_avg_agreements, color='purple', label=f'Moving Avg Agreement Rate (window={window_size})')
        plt.title('Expert Agreement Rate vs Training Progress')
        plt.xlabel('Episodes')
        plt.ylabel('Expert Agreement Rate')
        plt.grid(True)
        plt.legend()
        
        # Plot scatter of expert agreement rate vs performance
        plt.subplot(2, 1, 2)
        plt.scatter(moving_avg_agreements, moving_avg_win_rates, 
                   c=range(len(moving_avg_agreements)), cmap='viridis', alpha=0.7)
        plt.colorbar(label='Episodes')
        
        # Try to fit a trend line
        try:
            z = np.polyfit(moving_avg_agreements, moving_avg_win_rates, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(moving_avg_agreements), max(moving_avg_agreements), 100)
            plt.plot(x_range, p(x_range), "r--", alpha=0.8, label=f'Linear Trend: y={z[0]:.4f}x+{z[1]:.4f}')
        except:
            pass  # Skip trend line if fitting fails
        
        plt.title('Expert Agreement Rate vs Win Rate')
        plt.xlabel('Expert Agreement Rate (Moving Average)')
        plt.ylabel('Win Rate (Moving Average)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        # 设置输出目录
        if run_timestamp:
            output_dir = os.path.join('output', run_timestamp)
        else:
            output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save figure
        if episode is not None:
            plt.savefig(os.path.join(output_dir, f'expert_performance_relation_episode_{episode}.png'))
        else:
            plt.savefig(os.path.join(output_dir, f'expert_performance_relation_final.png'))
            
    finally:
        # 确保无论如何都关闭图表
        plt.close('all')