import argparse
from ai.adversarial_training import train_against_logic

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='训练RL智能体对抗Logic智能体')
    
    # 基本训练参数
    parser.add_argument('--episodes', type=int, default=50000, help='训练回合数')
    parser.add_argument('--save-interval', type=int, default=1000, help='模型保存间隔')
    parser.add_argument('--render', action='store_true', help='是否渲染游戏画面')
    parser.add_argument('--checkpoint', type=str, default=None, help='检查点文件路径，用于继续训练')
    
    # 专家策略学习参数
    expert_group = parser.add_mutually_exclusive_group()
    expert_group.add_argument('--use-expert', action='store_true', help='使用专家策略学习')
    expert_group.add_argument('--no-expert', action='store_false', dest='use_expert', help='不使用专家策略学习')
    parser.set_defaults(use_expert=True)
    parser.add_argument('--expert-reward', type=float, default=1.0, help='初始专家奖励值')
    parser.add_argument('--expert-decay', type=float, default=1.0, 
                        help='专家奖励衰减因子 (1.0表示完全按训练进度线性衰减，0表示不衰减)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 输出训练配置信息
    print("\n===== 训练配置 =====")
    print(f"训练回合数: {args.episodes}")
    print(f"模型保存间隔: {args.save_interval}")
    print(f"渲染游戏画面: {args.render}")
    print(f"检查点文件: {args.checkpoint if args.checkpoint else '无'}")
    
    if args.use_expert:
        print("\n===== 专家策略学习配置 =====")
        print(f"初始专家奖励值: {args.expert_reward}")
        print(f"专家奖励衰减因子: {args.expert_decay}")
        print("\n动作空间映射关系:")
        print("RL智能体: 0-停留, 1-前进, 2-后退, 3-左转, 4-右转, 5-开火")
        print("专家智能体: 0-停留, 1-前进, 2-左转, 3-右转, 4-开火 (无后退)")
        print("匹配规则: 0=0, 1=1, 3=2, 4=3, 5=4, 后退(2)永远不匹配")
    
    # 开始训练
    print("\n开始训练RL智能体对抗Logic智能体...")
    trained_agent = train_against_logic(
        episodes=args.episodes, 
        save_interval=args.save_interval, 
        render=args.render,
        checkpoint_path=args.checkpoint,
        use_expert_guidance=args.use_expert,
        expert_reward_init=args.expert_reward,
        expert_decay_factor=args.expert_decay
    )
    print("训练完成！")

if __name__ == "__main__":
    main()