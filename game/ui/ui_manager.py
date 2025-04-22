import pygame
import os
from typing import List, Dict, Optional

class UIManager:
    """UI管理器，负责渲染游戏界面"""
    
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        # 加载支持中文的字体
        font_path = os.path.join("assets", "fonts", "simhei.ttf")
        if not os.path.exists(font_path):
            # 如果没有找到指定字体，使用系统默认字体
            self.font_small = pygame.font.SysFont("SimHei", 24)
            self.font_medium = pygame.font.SysFont("SimHei", 36)
            self.font_large = pygame.font.SysFont("SimHei", 48)
        else:
            # 使用指定的字体文件
            self.font_small = pygame.font.Font(font_path, 24)
            self.font_medium = pygame.font.Font(font_path, 36)
            self.font_large = pygame.font.Font(font_path, 48)
    
    def render(self, game_state: Dict, tanks: List, game_over: bool, winner: Optional[int]):
        """渲染游戏UI"""
        # 渲染游戏状态信息
        self._render_game_info(game_state)
        
        # 渲染坦克信息
        self._render_tank_info(tanks)
        
        # 如果游戏结束，渲染游戏结束界面
        if game_over:
            self._render_game_over(winner)
    
    def _render_game_info(self, game_state: Dict):
        """渲染游戏状态信息"""
        # 渲染游戏时间
        time_text = f"时间: {game_state['time']:.1f}秒"
        time_surface = self.font_small.render(time_text, True, (255, 255, 255))
        self.screen.blit(time_surface, (10, 10))
        
        # 渲染击杀数
        kills_text = f"玩家1击杀: {game_state['player1_kills']} | 玩家2击杀: {game_state['player2_kills']}"
        kills_surface = self.font_small.render(kills_text, True, (255, 255, 255))
        self.screen.blit(kills_surface, (10, 40))
        
        # 渲染命中数
        hits_text = f"玩家1命中: {game_state['player1_hits']} | 玩家2命中: {game_state['player2_hits']}"
        hits_surface = self.font_small.render(hits_text, True, (255, 255, 255))
        self.screen.blit(hits_surface, (10, 70))
    
    def _render_tank_info(self, tanks: List):
        """渲染坦克信息"""
        for i, tank in enumerate(tanks):
            # 计算显示位置（右侧）
            x = self.screen.get_width() - 200
            y = 10 + i * 100
            
            # 渲染坦克ID和类型
            tank_text = f"坦克 {tank.player_id} ({tank.tank_type})"
            tank_surface = self.font_small.render(tank_text, True, (255, 255, 255))
            self.screen.blit(tank_surface, (x, y))
            
            # 渲染坦克生命值
            health_text = f"生命值: {tank.health:.1f}/{tank.max_health:.1f}"
            health_surface = self.font_small.render(health_text, True, (0, 255, 0))
            self.screen.blit(health_surface, (x, y + 25))
            
            # 渲染坦克弹药
            ammo_text = f"弹药: {tank.ammo}"
            ammo_surface = self.font_small.render(ammo_text, True, (255, 255, 0))
            self.screen.blit(ammo_surface, (x, y + 50))
            
            # 渲染装填进度
            reload_text = f"装填: {tank.reload_progress * 100:.0f}%"
            reload_surface = self.font_small.render(reload_text, True, (0, 191, 255))
            self.screen.blit(reload_surface, (x, y + 75))
    
    def _render_game_over(self, winner: Optional[int]):
        """渲染游戏结束界面"""
        # 创建半透明覆盖层
        overlay = pygame.Surface((self.screen.get_width(), self.screen.get_height()), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))  # 黑色半透明
        self.screen.blit(overlay, (0, 0))
        
        # 渲染游戏结束文本
        game_over_text = "游戏结束"
        game_over_surface = self.font_large.render(game_over_text, True, (255, 0, 0))
        game_over_rect = game_over_surface.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2 - 50))
        self.screen.blit(game_over_surface, game_over_rect)
        
        # 渲染胜利者文本
        if winner is not None:
            if winner > 0:
                winner_text = f"玩家 {winner} 胜利!"
            else:
                winner_text = "平局!"
            winner_surface = self.font_medium.render(winner_text, True, (255, 255, 0))
            winner_rect = winner_surface.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2))
            self.screen.blit(winner_surface, winner_rect)
        
        # 渲染重新开始提示
        restart_text = "按 R 键重新开始"
        restart_surface = self.font_small.render(restart_text, True, (255, 255, 255))
        restart_rect = restart_surface.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2 + 50))
        self.screen.blit(restart_surface, restart_rect)