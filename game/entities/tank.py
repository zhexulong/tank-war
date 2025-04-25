import pygame
import math
import random
from typing import List, Tuple, Dict, Optional

from game.config import TANK_TYPES, SCREEN_WIDTH, SCREEN_HEIGHT, TERRAIN_TYPES, TILE_SIZE, TERRAIN_COLORS
from game.entities.bullet import Bullet

class Tank:
    """坦克实体类，包含坦克的属性和行为"""
    
    def __init__(self, position: Tuple[int, int], tank_type: str, direction: float, is_player: bool = False, player_id: int = 0):
        # 基本属性
        self.position = position  # 坦克位置 (x, y)
        self.tank_type = tank_type  # 坦克类型 (light, medium, heavy)
        self.direction = direction  # 坦克朝向（角度，0-359）
        self.turret_direction = direction  # 炮塔朝向
        self.is_player = is_player  # 是否为玩家控制
        self.player_id = player_id  # 玩家ID（1或2）
        
        # 随机化坦克参数
        self.randomize_parameters()
        
        # 控制相关
        self.moving_forward = False
        self.moving_backward = False
        self.rotating_left = False
        self.rotating_right = False
        self.rotating_turret_left = False
        self.rotating_turret_right = False
        self.firing = False
        
        # 射击相关
        self.reload_time = 0  # 当前重新装填时间
        self.reload_progress = 1.0  # 装填进度（0-1）
        self.ammo = 10  # 弹药数量
        
        # AI相关
        self.ai_state = "idle"  # AI状态：idle, patrol, chase, attack
        self.ai_target = None  # AI目标
        self.ai_path = []  # AI路径
        self.ai_timer = 0  # AI计时器
        
        # 视觉表现 - 3x3像素风格
        self.size = (TILE_SIZE, TILE_SIZE)  # 坦克大小设为一个完整的格子大小
        self.color = TANK_TYPES[tank_type]["color"]  # 坦克主体颜色
        self.turret_length = self.size[0] // 3  # 炮管长度设为格子大小的1/3
    
    def randomize_parameters(self):
        """初始化坦克参数"""
        tank_config = TANK_TYPES[self.tank_type]
        
        # 根据坦克类型设置固定速度
        self.speed = tank_config["speed_range"][0]
        
        # 根据坦克类型设置固定生命值
        self.max_health = tank_config["health_range"][0]
        self.health = self.max_health
        
        # 根据坦克类型设置固定伤害
        self.damage = tank_config["damage_range"][0]
        
        # 根据坦克类型设置固定装填时间
        self.max_reload_time = tank_config["reload_time_range"][0] * 60  # 转换为帧数
    
    def handle_event(self, event: pygame.event.Event):
        """处理玩家输入事件"""
        if not self.is_player:
            return
        
        # 根据玩家ID确定控制键
        if self.player_id == 1:
            # 玩家1控制键 (WASD + QE + 空格)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    self.moving_forward = True
                elif event.key == pygame.K_s:
                    self.moving_backward = True
                elif event.key == pygame.K_a:
                    self.rotating_left = True
                elif event.key == pygame.K_d:
                    self.rotating_right = True
                elif event.key == pygame.K_q:
                    self.rotating_turret_left = True
                elif event.key == pygame.K_e:
                    self.rotating_turret_right = True
                elif event.key == pygame.K_SPACE:
                    self.firing = True
            
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_w:
                    self.moving_forward = False
                elif event.key == pygame.K_s:
                    self.moving_backward = False
                elif event.key == pygame.K_a:
                    self.rotating_left = False
                elif event.key == pygame.K_d:
                    self.rotating_right = False
                elif event.key == pygame.K_q:
                    self.rotating_turret_left = False
                elif event.key == pygame.K_e:
                    self.rotating_turret_right = False
                elif event.key == pygame.K_SPACE:
                    self.firing = False
        
        elif self.player_id == 2:
            # 玩家2控制键 (箭头键 + 小键盘1,3 + 回车)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.moving_forward = True
                elif event.key == pygame.K_DOWN:
                    self.moving_backward = True
                elif event.key == pygame.K_LEFT:
                    self.rotating_left = True
                elif event.key == pygame.K_RIGHT:
                    self.rotating_right = True
                elif event.key == pygame.K_KP1:
                    self.rotating_turret_left = True
                elif event.key == pygame.K_KP3:
                    self.rotating_turret_right = True
                elif event.key == pygame.K_RETURN:
                    self.firing = True
            
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    self.moving_forward = False
                elif event.key == pygame.K_DOWN:
                    self.moving_backward = False
                elif event.key == pygame.K_LEFT:
                    self.rotating_left = False
                elif event.key == pygame.K_RIGHT:
                    self.rotating_right = False
                elif event.key == pygame.K_KP1:
                    self.rotating_turret_left = False
                elif event.key == pygame.K_KP3:
                    self.rotating_turret_right = False
                elif event.key == pygame.K_RETURN:
                    self.firing = False
    
    def update(self, game_map, tanks: List):
        """更新坦克状态"""
        # 如果是AI控制，更新AI行为
        if not self.is_player:
            self.update_ai(game_map, tanks)
        
        # 更新移动
        self.update_movement(game_map)
        
        # 更新炮塔旋转
        self.update_turret_rotation()
        
        # 更新射击
        bullet = self.update_firing()
        return bullet
    
    def update_movement(self, game_map):
        """更新坦克移动"""
        # 计算移动方向和距离
        move_x, move_y = 0, 0
        
        if self.moving_forward:
            # 根据坦克朝向计算前进方向
            speed = max(0.8, self.speed)  # 提高最小前进速度
            move_x = math.cos(math.radians(self.direction)) * speed
            move_y = math.sin(math.radians(self.direction)) * speed
        
        if self.moving_backward:
            # 根据坦克朝向计算后退方向
            speed = max(0.6, self.speed * 0.75)  # 提高最小后退速度并增加后退速度系数
            move_x = -math.cos(math.radians(self.direction)) * speed
            move_y = -math.sin(math.radians(self.direction)) * speed
        
        # 计算新位置
        new_x = self.position[0] + move_x
        new_y = self.position[1] + move_y
        
        # 检查边界碰撞和地形碰撞
        if (0 <= new_x <= SCREEN_WIDTH - self.size[0] and 
            0 <= new_y <= SCREEN_HEIGHT - self.size[1] and
            self.check_terrain_collision(game_map, (new_x, new_y))):
            # 更新位置，使用round避免浮点数精度问题
            self.position = (round(new_x), round(new_y))
        else:
            # 如果发生碰撞，尝试单轴移动
            if 0 <= new_x <= SCREEN_WIDTH - self.size[0] and self.check_terrain_collision(game_map, (new_x, self.position[1])):
                self.position = (round(new_x), self.position[1])
            if 0 <= new_y <= SCREEN_HEIGHT - self.size[1] and self.check_terrain_collision(game_map, (self.position[0], new_y)):
                self.position = (self.position[0], round(new_y))
        
        # 更新旋转
        rotation_speed = 3  # 固定旋转速度
        if self.rotating_left:
            self.direction = (self.direction - rotation_speed) % 360
        
        if self.rotating_right:
            self.direction = (self.direction + rotation_speed) % 360
    
    def update_turret_rotation(self):
        """更新炮塔旋转"""
        if self.rotating_turret_left:
            self.turret_direction = (self.turret_direction - 3) % 360
        
        if self.rotating_turret_right:
            self.turret_direction = (self.turret_direction + 3) % 360
    
    def update_firing(self):
        """更新射击状态"""
        # 更新装填进度
        if self.reload_time > 0:
            self.reload_time -= 1
            self.reload_progress = 1 - (self.reload_time / self.max_reload_time)
        
        # 处理射击
        if self.firing and self.reload_time <= 0 and self.ammo > 0:
            # 创建子弹
            bullet_position = (
                self.position[0] + self.size[0] / 2 + math.cos(math.radians(self.turret_direction)) * self.turret_length,
                self.position[1] + self.size[1] / 2 + math.sin(math.radians(self.turret_direction)) * self.turret_length
            )
            
            bullet = Bullet(bullet_position, self.turret_direction, self.damage, self)
            
            # 重置装填时间和减少弹药
            self.reload_time = self.max_reload_time
            self.reload_progress = 0
            self.ammo -= 1
            
            # 返回创建的子弹，由游戏管理器添加到子弹列表中
            return bullet
        
        return None
    
    def update_ai(self, game_map, tanks: List):
        """更新AI行为"""
        # 简单的AI行为：寻找最近的敌人并攻击
        enemy_tanks = [tank for tank in tanks if tank.player_id != self.player_id]
        
        if enemy_tanks:
            # 找到最近的敌人
            nearest_enemy = min(enemy_tanks, key=lambda t: self.distance_to(t.position))
            self.ai_target = nearest_enemy
            
            # 计算与目标的距离
            distance = self.distance_to(nearest_enemy.position)
            
            if distance > 200:  # 如果距离较远，追击敌人
                self.ai_state = "chase"
                self.chase_target(nearest_enemy)
            else:  # 如果距离较近，攻击敌人
                self.ai_state = "attack"
                self.attack_target(nearest_enemy)
        else:
            # 没有敌人，随机巡逻
            self.ai_state = "patrol"
            self.patrol(game_map)
    
    def chase_target(self, target):
        """追击目标"""
        # 计算朝向目标的角度
        target_angle = self.angle_to(target.position)
        
        # 调整坦克朝向
        angle_diff = (target_angle - self.direction) % 360
        if angle_diff < 180:
            self.rotating_right = True
            self.rotating_left = False
        else:
            self.rotating_left = True
            self.rotating_right = False
        
        # 如果角度差不大，前进
        if abs(angle_diff) < 30 or abs(angle_diff - 360) < 30:
            self.moving_forward = True
            self.moving_backward = False
        else:
            self.moving_forward = False
            self.moving_backward = False
        
        # 调整炮塔朝向目标
        turret_angle_diff = (target_angle - self.turret_direction) % 360
        if turret_angle_diff < 180:
            self.rotating_turret_right = True
            self.rotating_turret_left = False
        else:
            self.rotating_turret_left = True
            self.rotating_turret_right = False
    
    def attack_target(self, target):
        """攻击目标"""
        # 停止移动
        self.moving_forward = False
        self.moving_backward = False
        
        # 调整炮塔朝向目标
        target_angle = self.angle_to(target.position)
        turret_angle_diff = (target_angle - self.turret_direction) % 360
        
        if turret_angle_diff < 180:
            self.rotating_turret_right = True
            self.rotating_turret_left = False
        else:
            self.rotating_turret_left = True
            self.rotating_turret_right = False
        
        # 如果炮塔朝向接近目标，开火
        if abs(turret_angle_diff) < 10 or abs(turret_angle_diff - 360) < 10:
            self.firing = True
        else:
            self.firing = False
    
    def patrol(self, game_map):
        """随机巡逻"""
        # 每隔一段时间改变方向
        self.ai_timer += 1
        if self.ai_timer >= 60:  # 每60帧改变一次方向
            self.ai_timer = 0
            # 随机选择一个新方向
            self.direction = random.randint(0, 359)
            # 随机决定是否移动
            self.moving_forward = random.choice([True, False])
        
        # 如果遇到障碍物，改变方向
        new_x = self.position[0] + math.cos(math.radians(self.direction)) * self.speed
        new_y = self.position[1] + math.sin(math.radians(self.direction)) * self.speed
        
        if not (0 <= new_x <= SCREEN_WIDTH - self.size[0] and 0 <= new_y <= SCREEN_HEIGHT - self.size[1]) or \
           not self.check_terrain_collision(game_map, (new_x, new_y)):
            # 遇到障碍物，改变方向
            self.direction = (self.direction + 180) % 360
    
    def check_terrain_collision(self, game_map, position: Tuple[int, int]) -> bool:
        """检查是否与地形碰撞"""
        # 计算缩小后的碰撞体积（原大小的70%）
        collision_size = (int(self.size[0] * 0.7), int(self.size[1] * 0.7))
        offset = ((self.size[0] - collision_size[0]) // 2, (self.size[1] - collision_size[1]) // 2)
        
        # 获取缩小后的碰撞体积四个角的位置
        corners = [
            (position[0] + offset[0], position[1] + offset[1]),  # 左上
            (position[0] + offset[0] + collision_size[0], position[1] + offset[1]),  # 右上
            (position[0] + offset[0], position[1] + offset[1] + collision_size[1]),  # 左下
            (position[0] + offset[0] + collision_size[0], position[1] + offset[1] + collision_size[1])  # 右下
        ]
        
        # 检查每个角是否在可通过的地形上
        for corner in corners:
            tile_x = int(corner[0] / game_map.tile_size)
            tile_y = int(corner[1] / game_map.tile_size)
            
            # 确保坐标在地图范围内
            if 0 <= tile_x < game_map.width and 0 <= tile_y < game_map.height:
                tile_type = game_map.get_tile(tile_x, tile_y)
                
                # 检查是否是不可通过的地形
                if tile_type in [TERRAIN_TYPES["brick"], TERRAIN_TYPES["steel"], TERRAIN_TYPES["water"]]:
                    return False
            else:
                # 超出地图边界
                return False
        
        return True
    
    def take_damage(self, damage: float):
        """受到伤害"""
        self.health -= damage
        if self.health < 0:
            self.health = 0
    
    def distance_to(self, position: Tuple[int, int]) -> float:
        """计算到指定位置的距离"""
        return math.sqrt((self.position[0] - position[0]) ** 2 + (self.position[1] - position[1]) ** 2)
    
    def angle_to(self, position: Tuple[int, int]) -> float:
        """计算到指定位置的角度"""
        dx = position[0] - self.position[0]
        dy = position[1] - self.position[1]
        return (math.degrees(math.atan2(dy, dx)) + 360) % 360
    
    def render(self, screen: pygame.Surface):
        """渲染坦克"""
        # 加载坦克贴图
        try:
            # 加载坦克底座贴图
            base_surface = pygame.image.load('game/assets/sprites/tank_base.svg')
            base_surface = pygame.transform.scale(base_surface, self.size)
            base_surface = pygame.transform.rotate(base_surface, -self.direction)
            
            # 加载坦克炮塔贴图
            turret_surface = pygame.image.load('game/assets/sprites/tank_turret.svg')
            turret_surface = pygame.transform.scale(turret_surface, self.size)
            turret_surface = pygame.transform.rotate(turret_surface, -self.turret_direction)
            
            # 绘制坦克底座
            base_rect = base_surface.get_rect(center=(self.position[0] + self.size[0]/2, 
                                                     self.position[1] + self.size[1]/2))
            screen.blit(base_surface, base_rect)
            
            # 绘制坦克炮塔
            turret_rect = turret_surface.get_rect(center=(self.position[0] + self.size[0]/2, 
                                                         self.position[1] + self.size[1]/2))
            screen.blit(turret_surface, turret_rect)
            
        except Exception as e:
            # 如果加载贴图失败，使用原始的几何图形渲染
            tank_rect = pygame.Rect(self.position[0], self.position[1], self.size[0], self.size[1])
            pygame.draw.rect(screen, self.color, tank_rect)
            
            direction_end = (
                self.position[0] + self.size[0] / 2 + math.cos(math.radians(self.direction)) * (self.size[0] / 2),
                self.position[1] + self.size[1] / 2 + math.sin(math.radians(self.direction)) * (self.size[1] / 2)
            )
            pygame.draw.line(screen, (255, 255, 255), 
                             (self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2),
                             direction_end, 2)
            
            turret_end = (
                self.position[0] + self.size[0] / 2 + math.cos(math.radians(self.turret_direction)) * self.turret_length,
                self.position[1] + self.size[1] / 2 + math.sin(math.radians(self.turret_direction)) * self.turret_length
            )
            pygame.draw.line(screen, (0, 0, 0), 
                             (self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2),
                             turret_end, 3)
        
        # 绘制生命值条
        health_width = (self.size[0] * (self.health / self.max_health))
        health_rect = pygame.Rect(self.position[0], self.position[1] - 10, health_width, 5)
        pygame.draw.rect(screen, (0, 255, 0), health_rect)
        
        # 绘制装填进度条
        reload_width = (self.size[0] * self.reload_progress)
        reload_rect = pygame.Rect(self.position[0], self.position[1] - 5, reload_width, 5)
        pygame.draw.rect(screen, (0, 0, 255), reload_rect)
        
        # 绘制玩家ID
        font = pygame.font.SysFont(None, 20)
        text = font.render(str(self.player_id), True, (255, 255, 255))
        screen.blit(text, (self.position[0] + self.size[0] / 2 - 5, self.position[1] + self.size[1] / 2 - 5))