from ai.adversarial_training import train_against_logic

def main():
    # 设置训练参数
    episodes = 1000  # 训练回合数
    save_interval = 100  # 模型保存间隔
    render = True  # 是否渲染游戏画面
    
    # 开始训练
    print("开始训练RL智能体对抗Logic智能体...")
    trained_agent = train_against_logic(episodes=episodes, save_interval=save_interval, render=render)
    print("训练完成！")

if __name__ == "__main__":
    main()