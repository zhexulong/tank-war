import random
import pygame
from typing import List, Tuple, Dict

from game.config import MAP_WIDTH, MAP_HEIGHT, TILE_SIZE, TERRAIN_TYPES, TERRAIN_COLORS

class GameMap:
    """游戏地图类，存储和管理地图数据"""
    
    def __init__(self, width: int, height: int, tile_size: int):
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.tiles = [[TERRAIN_TYPES["empty"] for _ in range(height)] for _ in range(width)]
    
    def get_tile(self, x: int, y: int) -> int:
        """获取指定位置的地形类型"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.tiles[x][y]
        return -1  # 超出边界
    
    def set_tile(self, x: int, y: int, tile_type: int):
        """设置指定位置的地形类型"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.tiles[x][y] = tile_type
    
    def render(self, screen: pygame.Surface):
        """渲染地图"""
        for x in range(self.width):
            for y in range(self.height):
                tile_type = self.tiles[x][y]
                if tile_type != TERRAIN_TYPES["empty"]:
                    # 绘制3x3的砖墙和钢墙
                    if tile_type in [TERRAIN_TYPES["brick"], TERRAIN_TYPES["steel"]]:
                        # 计算3x3网格的基准位置
                        base_x = x * self.tile_size
                        base_y = y * self.tile_size
                        
                        # 绘制3x3的小方块
                        for i in range(3):
                            for j in range(3):
                                block_rect = pygame.Rect(
                                    base_x + i * (self.tile_size // 3),
                                    base_y + j * (self.tile_size // 3),
                                    (self.tile_size // 3) - 1,
                                    (self.tile_size // 3) - 1
                                )
                                # 砖墙使用深红色，钢墙使用深灰色
                                color = (139, 0, 0) if tile_type == TERRAIN_TYPES["brick"] else (64, 64, 64)
                                pygame.draw.rect(screen, color, block_rect)
    
    def get_map_data(self) -> List[List[int]]:
        """获取地图数据，用于AI"""
        return [row[:] for row in self.tiles]  # 返回深拷贝

class MapGenerator:
    """地图生成器，负责生成随机地图"""
    
    def __init__(self):
        self.width = MAP_WIDTH
        self.height = MAP_HEIGHT
        self.tile_size = TILE_SIZE
    
    def generate_random_map(self) -> GameMap:
        """生成随机地图"""
        game_map = GameMap(self.width, self.height, self.tile_size)
        
        # 生成随机地形
        self._generate_terrain(game_map)
        
        # 确保地图中有安全区域（坦克生成点）
        self._ensure_spawn_areas(game_map)
        
        return game_map
    
    def _generate_terrain(self, game_map: GameMap):
        """生成随机地形"""
        # 随机生成砖墙
        self._generate_terrain_type(game_map, TERRAIN_TYPES["brick"], 0.08)  # 8%的概率是砖墙
        
        # 随机生成钢墙
        self._generate_terrain_type(game_map, TERRAIN_TYPES["steel"], 0.03)  # 3%的概率是钢墙
        
        # 生成一些障碍物集群
        self._generate_obstacle_clusters(game_map)
    
    def _generate_terrain_type(self, game_map: GameMap, terrain_type: int, probability: float):
        """生成指定类型的地形"""
        for x in range(game_map.width):
            for y in range(game_map.height):
                if random.random() < probability and game_map.get_tile(x, y) == TERRAIN_TYPES["empty"]:
                    game_map.set_tile(x, y, terrain_type)
    
    def _generate_obstacle_clusters(self, game_map: GameMap):
        """生成障碍物集群"""
        # 生成一些随机的障碍物集群 - 减少集群数量
        num_clusters = random.randint(2, 4)  # 原来是3-7个集群
        
        for _ in range(num_clusters):
            # 随机选择集群中心
            center_x = random.randint(0, game_map.width - 1)
            center_y = random.randint(0, game_map.height - 1)
            
            # 随机选择集群大小
            cluster_size = random.randint(3, 7)
            
            # 随机选择地形类型
            terrain_type = random.choice([TERRAIN_TYPES["brick"], TERRAIN_TYPES["steel"]])
            
            # 在中心周围生成障碍物
            for dx in range(-cluster_size // 2, cluster_size // 2 + 1):
                for dy in range(-cluster_size // 2, cluster_size // 2 + 1):
                    x, y = center_x + dx, center_y + dy
                    
                    # 确保坐标在地图范围内
                    if 0 <= x < game_map.width and 0 <= y < game_map.height:
                        # 根据到中心的距离计算概率
                        distance = (dx ** 2 + dy ** 2) ** 0.5
                        probability = 1 - (distance / (cluster_size // 2))
                        
                        if random.random() < probability and game_map.get_tile(x, y) == TERRAIN_TYPES["empty"]:
                            game_map.set_tile(x, y, terrain_type)
    
    def _ensure_spawn_areas(self, game_map: GameMap):
        """确保地图中有安全的坦克生成区域"""
        # 清空左上角区域（玩家1生成点）
        spawn_radius = 3
        for x in range(spawn_radius):
            for y in range(spawn_radius):
                game_map.set_tile(x, y, TERRAIN_TYPES["empty"])
        
        # 清空右下角区域（玩家2生成点）
        for x in range(game_map.width - spawn_radius, game_map.width):
            for y in range(game_map.height - spawn_radius, game_map.height):
                game_map.set_tile(x, y, TERRAIN_TYPES["empty"])