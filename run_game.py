#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
坦克世界大战游戏启动脚本

这个脚本提供了几种不同的游戏启动模式：
1. 人类玩家对战模式
2. 人类玩家vs AI模式
3. AI对战模式
4. AI训练模式
"""

import os
import sys
import argparse

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='坦克世界大战游戏')
    parser.add_argument('--game_type', type=str, default='normal',
                        choices=['normal', 'simplified'],
                        help='游戏类型：普通版或简化版')
    parser.add_argument('--mode', type=str, default='human_vs_ai',
                        choices=['human_vs_human', 'human_vs_ai', 'ai_vs_ai', 
                                'logic_ai_vs_human', 'logic_ai_vs_ai', 'train',
                                'human_vs_minimax', 'logic_ai_vs_minimax',
                                'logic_vs_naive_minimax'],
                        help='游戏模式')
    parser.add_argument('--agent', type=str, default=None,
                        help='AI智能体模型路径')
    parser.add_argument('--ai_type', type=str, default='dqn',
                        choices=['dqn', 'logic', 'minimax', 'naive_minimax'],
                        help='AI类型：dqn、logic、minimax或naive_minimax')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='训练回合数')
    parser.add_argument('--render', action='store_true',
                        help='是否渲染游戏画面')
    args = parser.parse_args()
    
    # 根据游戏类型选择启动不同版本
    if args.game_type == 'simplified':
        from game_simplified.game import main as start_simplified_game
        render_mode_value = 'human' if args.render else None
        
        if args.mode == 'human_vs_human':
            start_simplified_game(render_mode=render_mode_value)
        elif args.mode == 'human_vs_ai':
            start_simplified_game(ai_opponent=True, ai_type=args.ai_type, agent_path=args.agent, render_mode=render_mode_value)
        elif args.mode == 'ai_vs_ai':
            start_simplified_game(ai_opponent=True, ai_type=args.ai_type, second_ai_type=args.ai_type, render_mode=render_mode_value)
        elif args.mode == 'logic_ai_vs_human':
            start_simplified_game(ai_opponent=True, ai_type='logic', render_mode=render_mode_value)
        elif args.mode == 'logic_ai_vs_ai':
            start_simplified_game(ai_opponent=True, ai_type='logic', second_ai_type='logic', render_mode=render_mode_value)
        elif args.mode == 'human_vs_minimax':
            # Human is player 1, Minimax is player 2
            start_simplified_game(ai_opponent=True, ai_type='minimax', render_mode=render_mode_value)
        elif args.mode == 'minimax_vs_human':
            # Minimax is player 1, Human is player 2
            start_simplified_game(ai_opponent=True, ai_type='minimax', second_ai_type=None, render_mode=render_mode_value)
            # Swap player controls since minimax is player 1
            from game_simplified.game import SimplifiedGame
            SimplifiedGame.handle_input = lambda self: self._handle_input_p2_controls()
        elif args.mode == 'logic_ai_vs_minimax':
            # Both players are minimax agents
            start_simplified_game(ai_opponent=True, ai_type='logic', second_ai_type='minimax', render_mode=render_mode_value)
        elif args.mode == 'logic_vs_naive_minimax':
            # Compare minimax with naive minimax
            start_simplified_game(ai_opponent=True, ai_type='logic', second_ai_type='naive_minimax', render_mode=render_mode_value)
        elif args.mode == 'train':
            from ai.training import train_single_agent
            train_single_agent(episodes=args.episodes, render=args.render, simplified=True)
        return
    
    # 普通版游戏模式
    if args.mode == 'human_vs_human':
        # 人类玩家对战模式
        from main import main as start_game
        start_game()
    
    elif args.mode == 'human_vs_ai':
        # 人类玩家vs AI模式
        from main import main as start_game
        start_game(ai_opponent=True, ai_model_path=args.agent, ai_type=args.ai_type)
    
    elif args.mode == 'ai_vs_ai':
        # AI对战模式
        from ai.training import evaluate_agent
        evaluate_agent(args.agent, episodes=10, render=True, ai_type=args.ai_type)
    
    elif args.mode == 'logic_ai_vs_human':
        # 逻辑AI vs 人类模式
        from main import main as start_game
        start_game(ai_opponent=True, ai_type='logic')
    
    elif args.mode == 'logic_ai_vs_ai':
        # 逻辑AI对战模式
        from main import main as start_game
        start_game(ai_opponent=True, ai_type='logic', second_ai_type='logic')
    
    elif args.mode == 'human_vs_minimax':
        # Human vs Minimax mode (simplified only)
        sys.exit(1)
        
    elif args.mode == 'train':
        # AI训练模式
        from ai.training import train_single_agent, train_multi_agent
        train_single_agent(episodes=args.episodes, render=args.render)

if __name__ == '__main__':
    # 创建必要的目录
    directories = [
        "game", "game/entities", "game/map", "game/ui",
        "ai", "data", "assets", "models"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # 创建__init__.py文件
    init_files = [
        "ai/__init__.py",
        "data/__init__.py",
        "game/entities/__init__.py",
        "game/map/__init__.py",
        "game/ui/__init__.py"
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write(f"# {init_file.split('/')[-2]} 包初始化文件\n")
    
    main()
