import pygame
import math
import random
from typing import List, Dict, Tuple, Optional

from game.config import SCREEN_WIDTH, SCREEN_HEIGHT, FPS, TANK_TYPES, TERRAIN_TYPES
from game.entities.tank import Tank
from game.entities.bullet import Bullet
from game.map.map_generator import MapGenerator
from game.ui.ui_manager import UIManager
from game.game_manager import GameManager

class GameCore:
    """游戏核心类，负责处理游戏主循环、坦克控制和AI行为"""
    
    def __init__(self, game_manager: GameManager):
        self.game_manager = game_manager
        self.ai_difficulty = "medium"  # AI难度：easy, medium, hard
        self.ai_update_frequency = 10  # AI更新频率（帧数）
        self.frame_counter = 0
    
    def run_game_loop(self):
        """运行游戏主循环"""
        running = True
        clock = pygame.time.Clock()
        
        while running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.game_manager.handle_event(event)
            
            # 更新游戏状态
            self.update()
            
            # 渲染游戏
            self.game_manager.screen.fill((0, 0, 0))
            self.game_manager.render()
            pygame.display.flip()
            
            # 控制帧率
            clock.tick(FPS)
        
        return self.game_manager.winner
    
    def update(self):
        """更新游戏状态"""
        # 增加帧计数器
        self.frame_counter += 1
        
        # 更新游戏管理器
        self.game_manager.update()
        
        # 更新AI行为（按设定频率）
        if self.frame_counter % self.ai_update_frequency == 0:
            self.update_ai_behavior()
    
    def update_ai_behavior(self):
        """更新AI行为"""
        # 获取所有AI控制的坦克
        ai_tanks = [tank for tank in self.game_manager.tanks if not tank.is_player]
        
        for tank in ai_tanks:
            # 根据难度设置AI行为
            if self.ai_difficulty == "easy":
                self.update_easy_ai(tank)
            elif self.ai_difficulty == "medium":
                self.update_medium_ai(tank)
            elif self.ai_difficulty == "hard":
                self.update_hard_ai(tank)
    
    def update_easy_ai(self, tank: Tank):
        """简单AI行为：随机移动，偶尔射击"""
        # 随机移动
        if random.random() < 0.1:  # 10%概率改变移动状态
            tank.moving_forward = random.choice([True, False])
            tank.moving_backward = not tank.moving_forward if random.random() < 0.3 else False
        
        if random.random() < 0.1:  # 10%概率改变旋转状态
            tank.rotating_left = random.choice([True, False])
            tank.rotating_right = not tank.rotating_left if random.random() < 0.3 else False
        
        # 随机旋转炮塔
        if random.random() < 0.1:
            tank.rotating_turret_left = random.choice([True, False])
            tank.rotating_turret_right = not tank.rotating_turret_left if random.random() < 0.3 else False
        
        # 随机射击
        tank.firing = random.random() < 0.05  # 5%概率射击
    
    def update_medium_ai(self, tank: Tank):
        """中等AI行为：追踪最近的敌人，尝试射击"""
        # 获取所有敌方坦克
        enemy_tanks = [t for t in self.game_manager.tanks if t.player_id != tank.player_id]
        
        if enemy_tanks:
            # 找到最近的敌人
            nearest_enemy = min(enemy_tanks, key=lambda t: tank.distance_to(t.position))
            
            # 计算与目标的距离
            distance = tank.distance_to(nearest_enemy.position)
            
            # 根据距离决定行为
            if distance > 200:  # 如果距离较远，追击敌人
                self.chase_target(tank, nearest_enemy)
            else:  # 如果距离较近，攻击敌人
                self.attack_target(tank, nearest_enemy)
        else:
            # 没有敌人，随机巡逻
            tank.patrol(self.game_manager.current_map)
    
    def update_hard_ai(self, tank: Tank):
        """困难AI行为：智能追踪敌人，预测射击，利用掩体"""
        # 获取所有敌方坦克
        enemy_tanks = [t for t in self.game_manager.tanks if t.player_id != tank.player_id]
        
        if enemy_tanks:
            # 找到最近的敌人
            nearest_enemy = min(enemy_tanks, key=lambda t: tank.distance_to(t.position))
            
            # 计算与目标的距离
            distance = tank.distance_to(nearest_enemy.position)
            
            # 预测敌人移动
            predicted_position = self.predict_enemy_position(nearest_enemy)
            
            # 根据距离决定行为
            if distance > 300:  # 如果距离较远，智能追击敌人
                self.smart_chase(tank, nearest_enemy, predicted_position)
            elif distance > 100:  # 中等距离，寻找掩体并攻击
                self.find_cover_and_attack(tank, nearest_enemy, predicted_position)
            else:  # 如果距离较近，精确攻击敌人
                self.precise_attack(tank, nearest_enemy, predicted_position)
        else:
            # 没有敌人，智能巡逻
            self.smart_patrol(tank)
    
    def chase_target(self, tank: Tank, target: Tank):
        """追击目标"""
        # 计算朝向目标的角度
        target_angle = tank.angle_to(target.position)
        
        # 调整坦克朝向
        angle_diff = (target_angle - tank.direction) % 360
        if angle_diff < 180:
            tank.rotating_right = True
            tank.rotating_left = False
        else:
            tank.rotating_left = True
            tank.rotating_right = False
        
        # 如果角度差不大，前进
        if abs(angle_diff) < 30 or abs(angle_diff - 360) < 30:
            tank.moving_forward = True
            tank.moving_backward = False
        else:
            tank.moving_forward = False
            tank.moving_backward = False
        
        # 调整炮塔朝向目标
        turret_angle_diff = (target_angle - tank.turret_direction) % 360
        if turret_angle_diff < 180:
            tank.rotating_turret_right = True
            tank.rotating_turret_left = False
        else:
            tank.rotating_turret_left = True
            tank.rotating_turret_right = False
    
    def attack_target(self, tank: Tank, target: Tank):
        """攻击目标"""
        # 停止移动
        tank.moving_forward = False
        tank.moving_backward = False
        
        # 调整炮塔朝向目标
        target_angle = tank.angle_to(target.position)
        turret_angle_diff = (target_angle - tank.turret_direction) % 360
        
        if turret_angle_diff < 180:
            tank.rotating_turret_right = True
            tank.rotating_turret_left = False
        else:
            tank.rotating_turret_left = True
            tank.rotating_turret_right = False
        
        # 如果炮塔朝向接近目标，开火
        if abs(turret_angle_diff) < 10 or abs(turret_angle_diff - 360) < 10:
            tank.firing = True
        else:
            tank.firing = False
    
    def predict_enemy_position(self, enemy: Tank) -> Tuple[float, float]:
        """预测敌人未来位置"""
        # 简单预测：根据敌人当前移动状态预测未来位置
        future_steps = 10  # 预测未来10帧
        
        # 如果敌人正在移动，预测其未来位置
        if enemy.moving_forward:
            dx = math.cos(math.radians(enemy.direction)) * enemy.speed * future_steps
            dy = math.sin(math.radians(enemy.direction)) * enemy.speed * future_steps
            return (enemy.position[0] + dx, enemy.position[1] + dy)
        elif enemy.moving_backward:
            dx = -math.cos(math.radians(enemy.direction)) * (enemy.speed / 2) * future_steps
            dy = -math.sin(math.radians(enemy.direction)) * (enemy.speed / 2) * future_steps
            return (enemy.position[0] + dx, enemy.position[1] + dy)
        
        # 如果敌人不动，返回当前位置
        return enemy.position
    
    def smart_chase(self, tank: Tank, target: Tank, predicted_position: Tuple[float, float]):
        """智能追击目标"""
        # 计算朝向预测位置的角度
        target_angle = tank.angle_to(predicted_position)
        
        # 调整坦克朝向
        angle_diff = (target_angle - tank.direction) % 360
        if angle_diff < 180:
            tank.rotating_right = True
            tank.rotating_left = False
        else:
            tank.rotating_left = True
            tank.rotating_right = False
        
        # 如果角度差不大，前进
        if abs(angle_diff) < 20 or abs(angle_diff - 360) < 20:
            tank.moving_forward = True
            tank.moving_backward = False
        else:
            tank.moving_forward = False
            tank.moving_backward = False
        
        # 调整炮塔朝向目标当前位置
        current_target_angle = tank.angle_to(target.position)
        turret_angle_diff = (current_target_angle - tank.turret_direction) % 360
        if turret_angle_diff < 180:
            tank.rotating_turret_right = True
            tank.rotating_turret_left = False
        else:
            tank.rotating_turret_left = True
            tank.rotating_turret_right = False
        
        # 如果炮塔朝向接近目标，并且距离适中，开火
        if (abs(turret_angle_diff) < 5 or abs(turret_angle_diff - 360) < 5) and \
           100 < tank.distance_to(target.position) < 300:
            tank.firing = True
        else:
            tank.firing = False
    
    def find_cover_and_attack(self, tank: Tank, target: Tank, predicted_position: Tuple[float, float]):
        """寻找掩体并攻击"""
        # 简化实现：在敌人和自己之间保持一定距离，并利用地形
        distance = tank.distance_to(target.position)
        
        # 如果距离太近，后退
        if distance < 150:
            tank.moving_backward = True
            tank.moving_forward = False
        else:
            tank.moving_backward = False
            
            # 尝试找到一个有利的射击位置
            target_angle = tank.angle_to(target.position)
            perpendicular_angle1 = (target_angle + 90) % 360
            perpendicular_angle2 = (target_angle - 90) % 360
            
            # 选择一个垂直方向移动，以便侧面攻击
            if self.frame_counter % 100 < 50:  # 每100帧切换一次方向
                tank.direction = perpendicular_angle1
            else:
                tank.direction = perpendicular_angle2
            
            tank.moving_forward = True
        
        # 调整炮塔朝向预测位置
        predicted_angle = tank.angle_to(predicted_position)
        turret_angle_diff = (predicted_angle - tank.turret_direction) % 360
        if turret_angle_diff < 180:
            tank.rotating_turret_right = True
            tank.rotating_turret_left = False
        else:
            tank.rotating_turret_left = True
            tank.rotating_turret_right = False
        
        # 如果炮塔朝向接近预测位置，开火
        if abs(turret_angle_diff) < 8 or abs(turret_angle_diff - 360) < 8:
            tank.firing = True
        else:
            tank.firing = False
    
    def precise_attack(self, tank: Tank, target: Tank, predicted_position: Tuple[float, float]):
        """精确攻击目标"""
        # 计算与目标的距离
        distance = tank.distance_to(target.position)
        
        # 保持适当距离
        if distance < 100:
            # 后退
            tank.moving_backward = True
            tank.moving_forward = False
        elif distance > 150:
            # 前进
            tank.moving_forward = True
            tank.moving_backward = False
        else:
            # 停止移动
            tank.moving_forward = False
            tank.moving_backward = False
        
        # 调整炮塔朝向预测位置
        predicted_angle = tank.angle_to(predicted_position)
        turret_angle_diff = (predicted_angle - tank.turret_direction) % 360
        if turret_angle_diff < 180:
            tank.rotating_turret_right = True
            tank.rotating_turret_left = False
        else:
            tank.rotating_turret_left = True
            tank.rotating_turret_right = False
        
        # 如果炮塔朝向非常接近预测位置，开火
        if abs(turret_angle_diff) < 5 or abs(turret_angle_diff - 360) < 5:
            tank.firing = True
        else:
            tank.firing = False
    
    def smart_patrol(self, tank: Tank):
        """智能巡逻"""
        # 每隔一段时间改变方向
        if self.frame_counter % 120 == 0:  # 每120帧改变一次方向
            # 随机选择一个新方向
            tank.direction = random.randint(0, 359)
            # 随机决定是否移动
            tank.moving_forward = random.choice([True, False])
        
        # 检查前方是否有障碍物
        look_ahead_distance = 50
        look_ahead_x = tank.position[0] + math.cos(math.radians(tank.direction)) * look_ahead_distance
        look_ahead_y = tank.position[1] + math.sin(math.radians(tank.direction)) * look_ahead_distance
        
        # 如果前方有障碍物或即将超出边界，改变方向
        if not (0 <= look_ahead_x <= SCREEN_WIDTH and 0 <= look_ahead_y <= SCREEN_HEIGHT) or \
           not tank.check_terrain_collision(self.game_manager.current_map, (look_ahead_x, look_ahead_y)):
            # 改变方向
            tank.direction = (tank.direction + random.randint(90, 270)) % 360
        
        # 随机旋转炮塔，保持警戒
        if self.frame_counter % 60 == 0:  # 每60帧改变一次炮塔方向
            tank.turret_direction = random.randint(0, 359)
        
        # 偶尔射击，增加游戏趣味性
        tank.firing = random.random() < 0.01  # 1%概率射击
    
    def set_ai_difficulty(self, difficulty: str):
        """设置AI难度"""
        if difficulty in ["easy", "medium", "hard"]:
            self.ai_difficulty = difficulty
            
            # 根据难度调整AI更新频率
            if difficulty == "easy":
                self.ai_update_frequency = 15
            elif difficulty == "medium":
                self.ai_update_frequency = 10
            else:  # hard
                self.ai_update_frequency = 5