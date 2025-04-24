import pygame
import random
from typing import List, Dict, Tuple, Optional

from game.config import SCREEN_WIDTH, SCREEN_HEIGHT, FPS, RL_SETTINGS, TILE_SIZE, TERRAIN_TYPES, TERRAIN_COLORS
from game.entities.tank import Tank
from game.map.map_generator import MapGenerator
from game.ui.ui_manager import UIManager
from data.collector import DataCollector

class GameManager:
    """游戏管理器，负责管理游戏状态、实体和地图"""
    
    def __init__(self, screen: pygame.Surface, map_difficulty: float = 0.0):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.running = True
        self.game_over = False
        self.winner = None
        
        # 初始化地图生成器
        self.map_generator = MapGenerator()
        self.current_map = self.map_generator.generate_random_map(map_difficulty)
        
        # 初始化UI管理器
        self.ui_manager = UIManager(screen)
        
        # 初始化数据收集器
        self.data_collector = DataCollector()
        
        # 初始化坦克
        self.tanks = []
        self.bullets = []
        self.spawn_tanks()
        
        # 游戏计时器
        self.start_time = pygame.time.get_ticks()
        self.elapsed_time = 0
        
        # 游戏状态
        self.game_state = {
            "time": 0,
            "player1_kills": 0,
            "player2_kills": 0,
            "player1_hits": 0,
            "player2_hits": 0,
            "player1_damage_dealt": 0,
            "player2_damage_dealt": 0,
            "player1_damage_taken": 0,
            "player2_damage_taken": 0
        }
    
    def spawn_tanks(self):
        """生成坦克"""
        # 清空现有坦克
        self.tanks = []
        
        # 随机选择坦克类型
        tank_types = ["light", "medium", "heavy"]
        
        # 玩家1坦克（左上角）
        player1_type = random.choice(tank_types)
        player1_pos = self.find_valid_spawn_position(0, 0, SCREEN_WIDTH // 3, SCREEN_HEIGHT // 3)
        player1_tank = Tank(player1_pos, player1_type, 0, is_player=True, player_id=1)
        self.tanks.append(player1_tank)
        
        # 玩家2坦克（右下角）
        player2_type = random.choice(tank_types)
        player2_pos = self.find_valid_spawn_position(SCREEN_WIDTH * 2 // 3, SCREEN_HEIGHT * 2 // 3, 
                                                  SCREEN_WIDTH, SCREEN_HEIGHT)
        player2_tank = Tank(player2_pos, player2_type, 180, is_player=True, player_id=2)
        self.tanks.append(player2_tank)
    
    def find_valid_spawn_position(self, min_x: int, min_y: int, max_x: int, max_y: int) -> Tuple[int, int]:
        """在指定区域内找到有效的坦克生成位置"""
        # 改进实现：检查地形是否允许坦克生成，确保不会与墙重叠
        max_attempts = 20  # 最大尝试次数
        tank_size = (3 * TILE_SIZE // 4, 3 * TILE_SIZE // 4)  # 坦克尺寸
        
        for _ in range(max_attempts):
            # 随机生成位置
            pos_x = random.randint(min_x, max_x - tank_size[0])
            pos_y = random.randint(min_y, max_y - tank_size[1])
            position = (pos_x, pos_y)
            
            # 检查坦克四个角是否与地形碰撞
            corners = [
                position,  # 左上
                (position[0] + tank_size[0], position[1]),  # 右上
                (position[0], position[1] + tank_size[1]),  # 左下
                (position[0] + tank_size[0], position[1] + tank_size[1])  # 右下
            ]
            
            valid_position = True
            for corner in corners:
                tile_x = int(corner[0] / TILE_SIZE)
                tile_y = int(corner[1] / TILE_SIZE)
                
                # 确保坐标在地图范围内
                if 0 <= tile_x < self.current_map.width and 0 <= tile_y < self.current_map.height:
                    tile_type = self.current_map.get_tile(tile_x, tile_y)
                    
                    # 检查是否是不可通过的地形
                    if tile_type in [TERRAIN_TYPES["brick"], TERRAIN_TYPES["steel"], TERRAIN_TYPES["water"]]:
                        valid_position = False
                        break
                else:
                    # 超出地图边界
                    valid_position = False
                    break
            
            if valid_position:
                return position
        
        # 如果尝试多次仍找不到有效位置，清除该区域的障碍物
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        tile_center_x = center_x // TILE_SIZE
        tile_center_y = center_y // TILE_SIZE
        
        # 清除中心点周围3x3区域的障碍物
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                tx = tile_center_x + dx
                ty = tile_center_y + dy
                if 0 <= tx < self.current_map.width and 0 <= ty < self.current_map.height:
                    self.current_map.set_tile(tx, ty, TERRAIN_TYPES["empty"])
        
        return (center_x, center_y)
    
    def handle_event(self, event: pygame.event.Event):
        """处理游戏事件"""
        # 处理游戏结束重新开始
        if self.game_over and event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            self.reset_game()
            return
        
        # 将事件传递给坦克处理
        for tank in self.tanks:
            if tank.is_player:
                tank.handle_event(event)
    
    def update(self):
        """更新游戏状态"""
        if self.game_over:
            return
        
        # 更新游戏时间
        current_time = pygame.time.get_ticks()
        self.elapsed_time = (current_time - self.start_time) / 1000  # 转换为秒
        self.game_state["time"] = self.elapsed_time
        
        # 更新坦克
        for tank in self.tanks:
            # 更新坦克状态并检查是否创建了新子弹
            bullet = tank.update(self.current_map, self.tanks)
            
            # 如果创建了新子弹，添加到子弹列表中
            if bullet:
                self.bullets.append(bullet)
        
        # 更新子弹
        for bullet in self.bullets[:]:  # 使用副本进行迭代，以便安全删除
            bullet.update()
            
            # 检查子弹是否击中坦克
            for tank in self.tanks:
                if bullet.owner != tank and bullet.check_collision(tank):
                    # 记录击中数据
                    if bullet.owner.player_id == 1:
                        self.game_state["player1_hits"] += 1
                        self.game_state["player1_damage_dealt"] += bullet.damage
                        self.game_state["player2_damage_taken"] += bullet.damage
                    else:
                        self.game_state["player2_hits"] += 1
                        self.game_state["player2_damage_dealt"] += bullet.damage
                        self.game_state["player1_damage_taken"] += bullet.damage
                    
                    # 应用伤害
                    tank.take_damage(bullet.damage)
                    
                    # 检查坦克是否被摧毁
                    if tank.health <= 0:
                        if bullet.owner.player_id == 1:
                            self.game_state["player1_kills"] += 1
                        else:
                            self.game_state["player2_kills"] += 1
                        
                        # 移除坦克
                        self.tanks.remove(tank)
                        
                        # 检查游戏是否结束
                        if len(self.tanks) == 1:
                            self.game_over = True
                            self.winner = self.tanks[0].player_id
                            self.collect_game_data()
                    
                    # 移除子弹
                    self.bullets.remove(bullet)
                    break
            
            # 检查子弹是否击中地形
            if not bullet.check_terrain_collision(self.current_map):
                # 移除子弹
                if bullet in self.bullets:
                    self.bullets.remove(bullet)
        
        # 检查游戏是否超时
        if self.elapsed_time >= RL_SETTINGS["max_steps"] / FPS:
            self.game_over = True
            # 根据剩余血量决定胜者
            if len(self.tanks) > 1:
                if self.tanks[0].health > self.tanks[1].health:
                    self.winner = self.tanks[0].player_id
                elif self.tanks[0].health < self.tanks[1].health:
                    self.winner = self.tanks[1].player_id
                else:
                    self.winner = 0  # 平局
            self.collect_game_data()
    
    def render(self):
        """渲染游戏"""
        # 渲染地图
        self.current_map.render(self.screen)
        
        # 渲染子弹
        for bullet in self.bullets:
            bullet.render(self.screen)
        
        # 渲染坦克
        for tank in self.tanks:
            tank.render(self.screen)
        
        # 渲染UI
        self.ui_manager.render(self.game_state, self.tanks, self.game_over, self.winner)
    
    def reset_game(self, map_difficulty: float = 0.0):
        """重置游戏"""
        # 重新生成地图
        self.current_map = self.map_generator.generate_random_map(map_difficulty)
        
        # 重新生成坦克
        self.spawn_tanks()
        
        # 清空子弹
        self.bullets = []
        
        # 重置游戏状态
        self.game_over = False
        self.winner = None
        self.start_time = pygame.time.get_ticks()
        
        # 重置游戏数据
        self.game_state = {
            "time": 0,
            "player1_kills": 0,
            "player2_kills": 0,
            "player1_hits": 0,
            "player2_hits": 0,
            "player1_damage_dealt": 0,
            "player2_damage_dealt": 0,
            "player1_damage_taken": 0,
            "player2_damage_taken": 0
        }
    
    def collect_game_data(self):
        """收集游戏数据"""
        game_data = {
            "time": self.elapsed_time,
            "winner": self.winner,
            "player1_kills": self.game_state["player1_kills"],
            "player2_kills": self.game_state["player2_kills"],
            "player1_hits": self.game_state["player1_hits"],
            "player2_hits": self.game_state["player2_hits"],
            "player1_damage_dealt": self.game_state["player1_damage_dealt"],
            "player2_damage_dealt": self.game_state["player2_damage_dealt"],
            "player1_damage_taken": self.game_state["player1_damage_taken"],
            "player2_damage_taken": self.game_state["player2_damage_taken"]
        }
        
        self.data_collector.collect_data(game_data)
    
    def get_player_tank(self, player_id: int) -> Optional[Tank]:
        """获取指定玩家的坦克"""
        for tank in self.tanks:
            if tank.player_id == player_id:
                return tank
        return None
    
    def get_game_state_for_rl(self) -> Dict:
        """获取用于强化学习的游戏状态"""
        # 构建状态字典，包含坦克位置、朝向、生命值等信息
        state = {
            "tanks": [],
            "map": self.current_map.get_map_data(),
            "bullets": []
        }
        
        for tank in self.tanks:
            tank_data = {
                "id": tank.player_id,
                "position": tank.position,
                "direction": tank.direction,
                "health": tank.health,
                "ammo": tank.ammo,
                "reload_progress": tank.reload_progress
            }
            state["tanks"].append(tank_data)
        
        for bullet in self.bullets:
            bullet_data = {
                "position": bullet.position,
                "direction": bullet.direction,
                "owner_id": bullet.owner.player_id
            }
            state["bullets"].append(bullet_data)
        
        return state