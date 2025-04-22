import os
import sys
import pygame
from pygame.locals import *

# 添加项目路径到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入游戏模块
from game.game_manager import GameManager
from game.config import SCREEN_WIDTH, SCREEN_HEIGHT, TITLE, FPS

def main():
    # 初始化pygame
    pygame.init()
    pygame.display.set_caption(TITLE)
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    # 创建游戏管理器
    game_manager = GameManager(screen)
    
    # 主游戏循环
    running = True
    while running:
        # 处理事件
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            game_manager.handle_event(event)
        
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