import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def plot_training_curves(rewards, losses, win_rates, episode_lengths, moving_avg_reward=None, moving_avg_loss=None, moving_avg_win_rate=None, episode=None):
    """绘制训练曲线"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建输出目录
    os.makedirs('output', exist_ok=True)
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
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
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    if episode is not None:
        plt.savefig(f'output/training_curves_episode_{episode}.png')
    else:
        plt.savefig('output/training_curves_final.png')
    
    # 关闭图表
    plt.close()