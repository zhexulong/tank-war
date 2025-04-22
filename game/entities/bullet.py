import pygame
import math
from typing import Tuple, Optional

from game.config import TERRAIN_TYPES

class Bullet:
    """子弹实体类，处理子弹的移动、碰撞和伤害"""
    
    def __init__(self, position: Tuple[float, float], direction: float, damage: float, owner):
        self.position = position  # 子弹位置 (x, y)
        self.direction = direction  # 子弹方向（角度，0-359）
        self.damage = damage  # 子弹伤害
        self.owner = owner  # 发射子弹的坦克
        
        # 子弹属性
        self.speed = 10  # 提高子弹速度
        self.radius = 3  # 增大子弹半径
        self.color = (255, 255, 0)  # 调整子弹颜色为亮黄色
        
        # 计算子弹的速度向量
        self.velocity = (
            math.cos(math.radians(self.direction)) * self.speed,
            math.sin(math.radians(self.direction)) * self.speed
        )
    
    def update(self):
        """更新子弹位置"""
        # 更新位置
        self.position = (
            self.position[0] + self.velocity[0],
            self.position[1] + self.velocity[1]
        )
    
    def check_collision(self, tank) -> bool:
        """检查是否与坦克碰撞"""
        # 简单的矩形碰撞检测
        bullet_rect = pygame.Rect(
            self.position[0] - self.radius,
            self.position[1] - self.radius,
            self.radius * 2,
            self.radius * 2
        )
        
        tank_rect = pygame.Rect(
            tank.position[0],
            tank.position[1],
            tank.size[0],
            tank.size[1]
        )
        
        return bullet_rect.colliderect(tank_rect)
    
    def check_terrain_collision(self, game_map) -> bool:
        """检查是否与地形碰撞，返回是否继续存在"""
        # 检查是否超出屏幕边界
        if (self.position[0] < 0 or 
            self.position[0] > game_map.width * game_map.tile_size or
            self.position[1] < 0 or 
            self.position[1] > game_map.height * game_map.tile_size):
            return False
        
        # 获取子弹所在的地图格子
        tile_x = int(self.position[0] / game_map.tile_size)
        tile_y = int(self.position[1] / game_map.tile_size)
        
        # 确保坐标在地图范围内
        if 0 <= tile_x < game_map.width and 0 <= tile_y < game_map.height:
            tile_type = game_map.get_tile(tile_x, tile_y)
            
            # 检查是否碰到砖墙或钢墙
            if tile_type == TERRAIN_TYPES["brick"]:
                # 砖墙可以被摧毁
                game_map.set_tile(tile_x, tile_y, TERRAIN_TYPES["empty"])
                return False
            elif tile_type == TERRAIN_TYPES["steel"]:
                # 钢墙不能被摧毁
                return False
        
        return True
    
    def render(self, screen: pygame.Surface):
        """渲染子弹"""
        # 绘制子弹主体
        pygame.draw.circle(screen, self.color, (int(self.position[0]), int(self.position[1])), self.radius)
