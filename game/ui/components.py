import pygame
from typing import Tuple, Callable, List, Dict, Optional

class Button:
    """按钮组件，用于处理点击事件"""
    
    def __init__(self, rect: pygame.Rect, text: str, callback: Callable, 
                 color: Tuple[int, int, int] = (100, 100, 100),
                 hover_color: Tuple[int, int, int] = (150, 150, 150),
                 text_color: Tuple[int, int, int] = (255, 255, 255),
                 font_size: int = 24):
        self.rect = rect
        self.text = text
        self.callback = callback
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.font = pygame.font.SysFont(None, font_size)
        self.hovered = False
    
    def update(self, mouse_pos: Tuple[int, int], mouse_clicked: bool) -> bool:
        """更新按钮状态，返回是否被点击"""
        self.hovered = self.rect.collidepoint(mouse_pos)
        
        if self.hovered and mouse_clicked:
            self.callback()
            return True
        return False
    
    def render(self, screen: pygame.Surface):
        """渲染按钮"""
        # 绘制按钮背景
        pygame.draw.rect(screen, self.hover_color if self.hovered else self.color, self.rect)
        pygame.draw.rect(screen, (0, 0, 0), self.rect, 2)  # 边框
        
        # 绘制按钮文本
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)


class ProgressBar:
    """进度条组件，用于显示生命值、装填进度等"""
    
    def __init__(self, rect: pygame.Rect, value: float = 1.0, max_value: float = 1.0,
                 color: Tuple[int, int, int] = (0, 255, 0),
                 background_color: Tuple[int, int, int] = (50, 50, 50),
                 border_color: Tuple[int, int, int] = (0, 0, 0)):
        self.rect = rect
        self.value = value
        self.max_value = max_value
        self.color = color
        self.background_color = background_color
        self.border_color = border_color
    
    def update(self, value: float, max_value: Optional[float] = None):
        """更新进度条值"""
        self.value = value
        if max_value is not None:
            self.max_value = max_value
    
    def render(self, screen: pygame.Surface):
        """渲染进度条"""
        # 绘制背景
        pygame.draw.rect(screen, self.background_color, self.rect)
        
        # 计算填充宽度
        fill_width = int(self.rect.width * (self.value / self.max_value))
        fill_rect = pygame.Rect(self.rect.x, self.rect.y, fill_width, self.rect.height)
        
        # 绘制填充部分
        pygame.draw.rect(screen, self.color, fill_rect)
        
        # 绘制边框
        pygame.draw.rect(screen, self.border_color, self.rect, 2)


class Menu:
    """菜单组件，用于显示游戏菜单"""
    
    def __init__(self, rect: pygame.Rect, title: str, 
                 background_color: Tuple[int, int, int] = (50, 50, 50, 200),
                 title_color: Tuple[int, int, int] = (255, 255, 255)):
        self.rect = rect
        self.title = title
        self.background_color = background_color
        self.title_color = title_color
        self.font_title = pygame.font.SysFont(None, 48)
        self.buttons = []
    
    def add_button(self, button: Button):
        """添加按钮到菜单"""
        self.buttons.append(button)
    
    def update(self, mouse_pos: Tuple[int, int], mouse_clicked: bool) -> Optional[str]:
        """更新菜单状态，返回被点击的按钮文本"""
        for button in self.buttons:
            if button.update(mouse_pos, mouse_clicked):
                return button.text
        return None
    
    def render(self, screen: pygame.Surface):
        """渲染菜单"""
        # 创建半透明背景
        menu_surface = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        pygame.draw.rect(menu_surface, self.background_color, menu_surface.get_rect())
        
        # 绘制标题
        title_surface = self.font_title.render(self.title, True, self.title_color)
        title_rect = title_surface.get_rect(center=(self.rect.width // 2, 50))
        menu_surface.blit(title_surface, title_rect)
        
        # 绘制按钮
        for button in self.buttons:
            button.render(menu_surface)
        
        # 将菜单绘制到屏幕
        screen.blit(menu_surface, self.rect.topleft)


class StatusDisplay:
    """状态显示组件，用于显示游戏状态信息"""
    
    def __init__(self, rect: pygame.Rect, 
                 background_color: Tuple[int, int, int] = (0, 0, 0, 100),
                 text_color: Tuple[int, int, int] = (255, 255, 255)):
        self.rect = rect
        self.background_color = background_color
        self.text_color = text_color
        self.font = pygame.font.SysFont(None, 24)
        self.items = {}  # 存储状态项 {name: value}
    
    def update_item(self, name: str, value):
        """更新状态项"""
        self.items[name] = value
    
    def render(self, screen: pygame.Surface):
        """渲染状态显示"""
        # 创建半透明背景
        status_surface = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        pygame.draw.rect(status_surface, self.background_color, status_surface.get_rect())
        
        # 绘制状态项
        y_offset = 10
        for name, value in self.items.items():
            text = f"{name}: {value}"
            text_surface = self.font.render(text, True, self.text_color)
            status_surface.blit(text_surface, (10, y_offset))
            y_offset += 30
        
        # 将状态显示绘制到屏幕
        screen.blit(status_surface, self.rect.topleft)


class TankInfoPanel:
    """坦克信息面板，显示坦克的详细信息"""
    
    def __init__(self, rect: pygame.Rect, 
                 background_color: Tuple[int, int, int] = (0, 0, 0, 150),
                 text_color: Tuple[int, int, int] = (255, 255, 255)):
        self.rect = rect
        self.background_color = background_color
        self.text_color = text_color
        self.font_title = pygame.font.SysFont(None, 28)
        self.font_info = pygame.font.SysFont(None, 24)
        
        # 创建进度条
        bar_width = rect.width - 20
        self.health_bar = ProgressBar(
            pygame.Rect(10, 60, bar_width, 20),
            color=(0, 255, 0)
        )
        self.reload_bar = ProgressBar(
            pygame.Rect(10, 110, bar_width, 20),
            color=(0, 191, 255)
        )
    
    def update(self, tank):
        """更新坦克信息"""
        self.tank = tank
        self.health_bar.update(tank.health, tank.max_health)
        self.reload_bar.update(tank.reload_progress)
    
    def render(self, screen: pygame.Surface):
        """渲染坦克信息面板"""
        # 创建半透明背景
        panel_surface = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        pygame.draw.rect(panel_surface, self.background_color, panel_surface.get_rect())
        
        # 绘制坦克标题
        title_text = f"坦克 {self.tank.player_id} ({self.tank.tank_type})"
        title_surface = self.font_title.render(title_text, True, self.text_color)
        panel_surface.blit(title_surface, (10, 10))
        
        # 绘制生命值文本
        health_text = f"生命值: {self.tank.health:.1f}/{self.tank.max_health:.1f}"
        health_surface = self.font_info.render(health_text, True, self.text_color)
        panel_surface.blit(health_surface, (10, 40))
        
        # 绘制生命值进度条
        self.health_bar.render(panel_surface)
        
        # 绘制装填进度文本
        reload_text = f"装填进度: {self.tank.reload_progress * 100:.0f}%"
        reload_surface = self.font_info.render(reload_text, True, self.text_color)
        panel_surface.blit(reload_surface, (10, 90))
        
        # 绘制装填进度条
        self.reload_bar.render(panel_surface)
        
        # 绘制弹药信息
        ammo_text = f"弹药: {self.tank.ammo}"
        ammo_surface = self.font_info.render(ammo_text, True, self.text_color)
        panel_surface.blit(ammo_surface, (10, 140))
        
        # 绘制速度信息
        speed_text = f"速度: {self.tank.speed:.1f}"
        speed_surface = self.font_info.render(speed_text, True, self.text_color)
        panel_surface.blit(speed_surface, (10, 170))
        
        # 绘制伤害信息
        damage_text = f"伤害: {self.tank.damage:.1f}"
        damage_surface = self.font_info.render(damage_text, True, self.text_color)
        panel_surface.blit(damage_surface, (10, 200))
        
        # 将面板绘制到屏幕
        screen.blit(panel_surface, self.rect.topleft)