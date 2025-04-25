import os
import sys
import pygame
from pygame.locals import *

# 添加项目路径到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# 导入游戏模块
from game.game_manager import GameManager
from game.core import GameCore
from game.config import SCREEN_WIDTH, SCREEN_HEIGHT, TITLE, FPS


def main(ai_opponent=False, ai_model_path=None, ai_type='dqn', second_ai_type=None):
    """游戏主入口
    
    Args:
        ai_opponent: 是否启用AI对手
        ai_model_path: AI模型路径（仅DQN需要）
        ai_type: AI类型 ('dqn' 或 'logic')
        second_ai_type: 第二个AI类型（用于AI对战）
    """
    # 初始化pygame
    pygame.init()
    pygame.display.set_caption(TITLE)
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    
    # 创建游戏管理器
    game_manager = GameManager(screen)
    game_core = GameCore(game_manager)
    
    # 如果启用AI，创建AI接口
    if ai_opponent:
        from ai.interface import AIInterface
        ai_interface = AIInterface(game_manager, game_core)
        
        print(f"正在初始化AI - 类型: {ai_type}")
        # 初始化主AI（玩家2）
        ai_interface.load_agent(2, ai_model_path, ai_type)
        
        # 如果是AI对战模式，初始化第二个AI（玩家1）
        if second_ai_type:
            print(f"正在初始化第二个AI - 类型: {second_ai_type}")
            ai_interface.load_agent(1, None, second_ai_type)
            # 设置玩家1的坦克为AI控制
            for tank in game_manager.tanks:
                if tank.player_id == 1:
                    tank.is_player = False
    
    # 主游戏循环
    running = True
    clock = pygame.time.Clock()
    frame_count = 0
    
    while running:
        frame_count += 1
        # 处理事件
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            game_manager.handle_event(event)
        
        # 更新AI行为
        if ai_opponent and frame_count % 60 == 0:  # 每秒输出一次调试信息
            print("\nAI状态更新:")
            for tank in game_manager.tanks:
                print(f"坦克 {tank.player_id} - 位置: {tank.position}, 朝向: {tank.direction}, 生命值: {tank.health}")
            ai_interface.update_ai_controlled_tanks()
        else:
            if ai_opponent:
                ai_interface.update_ai_controlled_tanks()
        
        # 更新游戏状态
        game_manager.update()
        
        # 渲染游戏
        screen.fill((0, 0, 0))
        game_manager.render()
        pygame.display.flip()
        
        # 控制帧率
        clock.tick(FPS)
    
    # 退出游戏
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    # 创建必要的目录
    directories = [
        "game", "game/entities", "game/map", "game/ui",
        "ai", "data", "assets"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    main()