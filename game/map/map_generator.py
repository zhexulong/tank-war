import random
import pygame
from typing import List, Tuple, Dict

from game.config import MAP_WIDTH, MAP_HEIGHT, TILE_SIZE, TERRAIN_TYPES, TERRAIN_COLORS
from game.map.terrain_textures import TerrainTextures

class GameMap:
    """游戏地图类，存储和管理地图数据"""
    
    def __init__(self, width: int, height: int, tile_size: int):
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.tiles = [[TERRAIN_TYPES["empty"] for _ in range(height)] for _ in range(width)]
        self.terrain_textures = TerrainTextures()
    
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
                    # 获取对应的地形贴图
                    texture = None
                    if tile_type == TERRAIN_TYPES["brick"]:
                        texture = self.terrain_textures.get_texture("brick")
                    elif tile_type == TERRAIN_TYPES["steel"]:
                        texture = self.terrain_textures.get_texture("steel")
                    elif tile_type == TERRAIN_TYPES["water"]:
                        texture = self.terrain_textures.get_texture("water")
                    elif tile_type == TERRAIN_TYPES["grass"]:
                        texture = self.terrain_textures.get_texture("grass")
                    
                    if texture:
                        # 计算贴图位置
                        pos_x = x * self.tile_size
                        pos_y = y * self.tile_size
                        screen.blit(texture, (pos_x, pos_y))
    
    def get_map_data(self) -> List[List[int]]:
        """获取地图数据，用于AI"""
        return [row[:] for row in self.tiles]  # 返回深拷贝

class MapGenerator:
    """地图生成器，负责生成随机地图"""
    
    def __init__(self):
        self.width = MAP_WIDTH
        self.height = MAP_HEIGHT
        self.tile_size = TILE_SIZE

    def generate_random_map(self, difficulty: float = 0.0) -> GameMap:
        """生成随机地图，difficulty取值0~1，越大障碍越多"""
        game_map = GameMap(self.width, self.height, self.tile_size)
        self._generate_terrain(game_map, difficulty)
        self._ensure_spawn_areas(game_map)
        return game_map

    def _generate_terrain(self, game_map: GameMap, difficulty: float = 0.0):
        """生成随机地形，难度越高障碍越多"""
        # 增加障碍概率随难度线性变化
        brick_prob = 0.03 + 0.10 * difficulty  # 0.03~0.13
        steel_prob = 0.01 + 0.06 * difficulty  # 0.01~0.07
        self._generate_terrain_type(game_map, TERRAIN_TYPES["brick"], brick_prob)
        self._generate_terrain_type(game_map, TERRAIN_TYPES["steel"], steel_prob)
        self._generate_obstacle_clusters(game_map, difficulty)

    def _generate_terrain_type(self, game_map: GameMap, terrain_type: int, probability: float):
        """生成指定类型的地形"""
        for x in range(game_map.width):
            for y in range(game_map.height):
                if random.random() < probability and game_map.get_tile(x, y) == TERRAIN_TYPES["empty"]:
                    game_map.set_tile(x, y, terrain_type)
    
    def _generate_obstacle_clusters(self, game_map: GameMap, difficulty: float = 0.0):
        """生成障碍物集群，难度越高集群越多"""
        min_clusters = 1 + int(2 * difficulty)  # 1~3
        max_clusters = 3 + int(5 * difficulty)  # 3~8
        num_clusters = random.randint(min_clusters, max_clusters)
        
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