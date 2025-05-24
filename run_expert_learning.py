#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
专家策略学习训练脚本示例
"""

import subprocess

def main():
    # 专家策略学习示例 - 默认配置
    print("==========================================")
    print("开始使用默认专家策略学习训练...")
    print("==========================================")
    subprocess.run(["python", "train_script.py", 
                    "--use-expert", 
                    "--expert-reward", "1.0", 
                    "--expert-decay", "1.0"])
    
    # 非衰减的专家策略学习示例 - 高度依赖专家
    print("\n\n==========================================")
    print("开始使用非衰减专家策略学习训练...")
    print("非衰减将导致智能体持续依赖专家指导")
    print("==========================================")
    subprocess.run(["python", "train_script.py", 
                    "--use-expert", 
                    "--expert-reward", "1.0", 
                    "--expert-decay", "0.0"])
    
    # 不使用专家策略学习
    print("\n\n==========================================")
    print("开始不使用专家指导的标准强化学习训练...")
    print("==========================================")
    subprocess.run(["python", "train_script.py", 
                    "--no-expert"])

if __name__ == "__main__":
    main()
