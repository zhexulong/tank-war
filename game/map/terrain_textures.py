import pygame
import os
import random

class TerrainTextures:
    """地形贴图管理器，负责加载和管理地形贴图"""
    
    def __init__(self):
        self.textures = {}
        self._load_textures()
    
    def _load_textures(self):
        """加载地形贴图"""
        # 确保贴图目录存在
        textures_dir = os.path.join("assets", "textures", "terrain")
        os.makedirs(textures_dir, exist_ok=True)
        
        # 如果没有贴图文件，创建基础像素风格贴图
        self._create_pixel_textures()
    
    def _create_pixel_textures(self):
        """创建基础像素风格贴图"""
        # 砖墙贴图 - 3x3像素风格
        brick_surface = pygame.Surface((40, 40))
        brick_surface.fill((0, 0, 0))  # 黑色背景
        
        # 绘制3x3砖块图案
        brick_color = (139, 69, 19)  # 深棕色
        mortar_color = (169, 169, 169)  # 浅灰色
        
        for i in range(3):
            for j in range(3):
                # 绘制砖块
                block_rect = pygame.Rect(
                    i * 13 + 1,
                    j * 13 + 1,
                    11,
                    11
                )
                pygame.draw.rect(brick_surface, brick_color, block_rect)
                
                # 绘制砂浆纹理
                pygame.draw.line(brick_surface, mortar_color,
                               (i * 13, j * 13),
                               (i * 13 + 13, j * 13))
                pygame.draw.line(brick_surface, mortar_color,
                               (i * 13, j * 13),
                               (i * 13, j * 13 + 13))
        
        self.textures["brick"] = brick_surface
        
        # 钢墙贴图 - 金属质感
        steel_surface = pygame.Surface((40, 40))
        steel_surface.fill((0, 0, 0))  # 黑色背景
        
        # 主体金属色
        steel_color = (128, 128, 128)  # 中灰色
        highlight_color = (192, 192, 192)  # 亮灰色
        shadow_color = (64, 64, 64)  # 暗灰色
        
        # 绘制主体金属板
        main_rect = pygame.Rect(2, 2, 36, 36)
        pygame.draw.rect(steel_surface, steel_color, main_rect)
        
        # 绘制金属纹理
        for i in range(2):
            for j in range(2):
                # 绘制高光边缘
                highlight_rect = pygame.Rect(
                    i * 20 + 4,
                    j * 20 + 4,
                    16,
                    16
                )
                pygame.draw.rect(steel_surface, highlight_color, highlight_rect, 1)
                
                # 绘制阴影效果
                shadow_rect = pygame.Rect(
                    i * 20 + 5,
                    j * 20 + 5,
                    14,
                    14
                )
                pygame.draw.rect(steel_surface, shadow_color, shadow_rect, 1)
        
        self.textures["steel"] = steel_surface
        
        # 水面贴图 - 波纹效果
        water_surface = pygame.Surface((40, 40))
        water_surface.fill((0, 0, 0))  # 黑色背景
        
        # 水面颜色
        water_colors = [
            (0, 0, 139),  # 深蓝色
            (0, 0, 205),  # 中蓝色
            (0, 0, 255)   # 亮蓝色
        ]
        
        # 绘制波纹图案
        for i in range(4):
            wave_color = water_colors[i % len(water_colors)]
            wave_rect = pygame.Rect(0, i * 10, 40, 8)
            pygame.draw.rect(water_surface, wave_color, wave_rect)
        
        self.textures["water"] = water_surface
        
        # 草地贴图 - 随机点状图案
        grass_surface = pygame.Surface((40, 40))
        grass_surface.fill((34, 139, 34))  # 森林绿色背景
        
        # 随机添加深浅不一的草点
        grass_colors = [
            (0, 100, 0),    # 深绿色
            (50, 205, 50),  # 浅绿色
            (144, 238, 144) # 淡绿色
        ]
        
        for _ in range(30):
            color = grass_colors[random.randint(0, len(grass_colors) - 1)]
            x = random.randint(0, 39)
            y = random.randint(0, 39)
            grass_surface.set_at((x, y), color)
        
        self.textures["grass"] = grass_surface
    
    def get_texture(self, terrain_type: str) -> pygame.Surface:
        """获取指定类型的地形贴图"""
        return self.textures.get(terrain_type, None)