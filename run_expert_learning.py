#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
专家策略学习训练脚本示例
"""

import subprocess
import argparse

def run_training(algorithm='dqn'):
    """执行不同配置的训练任务"""
    # 获取算法命令行参数
    algorithm_arg = f"--algorithm {algorithm}"
    
    # 专家策略学习示例 - 默认配置
    print("==========================================")
    print(f"开始使用{algorithm.upper()}算法的默认专家策略学习训练...")
    print("==========================================")
    subprocess.run(["python", "train_script.py", 
                    algorithm_arg,
                    "--use-expert", 
                    "--expert-reward", "1.0", 
                    "--expert-decay", "1.0"])
    
    # 非衰减的专家策略学习示例 - 高度依赖专家
    print("\n\n==========================================")
    print(f"开始使用{algorithm.upper()}算法的非衰减专家策略学习训练...")
    print("非衰减将导致智能体持续依赖专家指导")
    print("==========================================")
    subprocess.run(["python", "train_script.py", 
                    algorithm_arg,
                    "--use-expert", 
                    "--expert-reward", "1.0", 
                    "--expert-decay", "0.0"])
    
    # 不使用专家策略学习
    print("\n\n==========================================")
    print(f"开始不使用专家指导的{algorithm.upper()}标准强化学习训练...")
    print("==========================================")
    subprocess.run(["python", "train_script.py", 
                    algorithm_arg,
                    "--no-expert"])

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='专家策略学习训练脚本示例')
    parser.add_argument('--algorithm', type=str, default='dqn', choices=['dqn', 'ppo'],
                        help='选择训练算法: dqn或ppo (默认: dqn)')
    args = parser.parse_args()
    
    # 执行指定算法的训练
    run_training(args.algorithm)

if __name__ == "__main__":
    main()
