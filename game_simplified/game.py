# 导入所需模块
import pygame
import random
from ai.simplified_interface import SimplifiedAIInterface  # 使用简化版AI接口
from game.core import GameCore

# 地形类型常量
EMPTY = 0
BRICK = 1
STEEL = 2
WATER = 3
GRASS = 4

class SimplifiedMap:
    """简化版地图类"""
    def __init__(self, width, height, tile_size):
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.grid = [[EMPTY for _ in range(width)] for _ in range(height)]
        self.obstacles = set()  # 存储障碍物位置
    
    def get_tile(self, x, y):
        """获取指定位置的地形类型"""
        if 0 <= x < self.width and 0 <= y < self.height:
            if (x, y) in self.obstacles:
                return BRICK
            return EMPTY
        return STEEL  # 地图边界视为钢墙
    
    def add_obstacle(self, x, y):
        """添加障碍物"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.obstacles.add((x, y))
    
    def remove_obstacle(self, x, y):
        """移除障碍物"""
        if (x, y) in self.obstacles:
            self.obstacles.remove((x, y))
    
    def is_obstacle(self, x, y):
        """检查位置是否有障碍物"""
        return (x, y) in self.obstacles

class SimplifiedTank:
    """简化版坦克类"""
    def __init__(self, x, y, direction, player_id):
        self.position = [x, y]
        self.direction = direction
        self.player_id = player_id
        self.health = 1
        self.is_player = True
    
    def update_position(self, new_pos):
        self.position[0] = new_pos[0]
        self.position[1] = new_pos[1]
    
    def update_direction(self, new_direction):
        self.direction = new_direction

class SimplifiedGame:
    def __init__(self, ai_opponent=False, ai_type='dqn', second_ai_type=None):
        pygame.init()
        
        # 游戏常量
        self.GRID_SIZE = 10  # 每个格子的像素大小
        self.MAP_SIZE = 16   # 地图大小(格子数)
        self.SCREEN_SIZE = self.GRID_SIZE * self.MAP_SIZE  # 屏幕大小
        
        # 初始化屏幕
        self.screen = pygame.display.set_mode((self.SCREEN_SIZE, self.SCREEN_SIZE))
        pygame.display.set_caption('简化版坦克大战')
        
        # 颜色定义
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (128, 128, 128)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        
        # 创建地图
        self.current_map = SimplifiedMap(self.MAP_SIZE, self.MAP_SIZE, self.GRID_SIZE)
        
        # 游戏状态
        self.running = True
        self.game_over = False
        self.winner = None
        
        # 游戏元素
        self.tanks = []
        self.bullets = []
        
        # AI相关设置
        self.ai_opponent = ai_opponent
        self.ai_type = ai_type
        self.second_ai_type = second_ai_type
        self.ai_interface = None
        
        # 初始化游戏
        self.init_game()
        
        # 如果启用AI，初始化AI接口
        if self.ai_opponent:
            self.setup_ai()
    
    def init_game(self):
        """初始化游戏状态"""
        # 重置坦克位置
        self.tanks = [
            SimplifiedTank(2, 2, 0, 1),  # 玩家1坦克
            SimplifiedTank(self.MAP_SIZE - 3, self.MAP_SIZE - 3, 2, 2)  # 玩家2坦克
        ]
        
        # 清空子弹
        self.bullets = []
        
        # 重置游戏状态
        self.game_over = False
        self.winner = None
        
        # 生成障碍物
        self.current_map.obstacles = self.generate_obstacles()
    
    def setup_ai(self):
        """设置AI系统"""
        game_core = GameCore(self)
        self.ai_interface = SimplifiedAIInterface(self, game_core)
        
        print(f"正在初始化AI - 类型: {self.ai_type}")
        # 初始化AI（玩家2）
        self.ai_interface.load_agent(2, None, self.ai_type)
        self.tanks[1].is_player = False
        
        # 如果是AI对战模式
        if self.second_ai_type:
            print(f"正在初始化第二个AI - 类型: {self.second_ai_type}")
            self.ai_interface.load_agent(1, None, self.second_ai_type)
            self.tanks[0].is_player = False  # 设置玩家1的坦克为AI控制

    def get_player_tank(self, player_id):
        """获取指定玩家的坦克"""
        for tank in self.tanks:
            if tank.player_id == player_id:
                return tank
        return None

    def get_game_state_for_rl(self):
        """获取用于强化学习的游戏状态"""
        return {
            'map': self.current_map.grid,  # 添加地图网格信息
            'tanks': [
                {'position': tank.position, 'direction': tank.direction, 'player_id': tank.player_id}
                for tank in self.tanks
            ],
            'bullets': self.bullets,
            'obstacles': list(self.current_map.obstacles),
            'game_over': self.game_over,
            'winner': self.winner
        }
    
    def get_state_shape(self):
        """获取游戏状态空间的形状"""
        return {
            'map': (self.MAP_SIZE, self.MAP_SIZE),  # 地图网格
            'tanks': (2, 4),  # 两辆坦克，每个坦克 [x,y,direction,player_id]
            'bullets': (10, 4),  # 最多10颗子弹，每个子弹 [x,y,direction,player_id]
        }
    
    def generate_obstacles(self, num_obstacles=50):
        obstacles = set()
        # 保护两个坦克的初始区域
        tank1_area = {(x, y) for x in range(self.tanks[0].position[0]-1, self.tanks[0].position[0]+2)
                            for y in range(self.tanks[0].position[1]-1, self.tanks[0].position[1]+2)}
        tank2_area = set()
        if self.ai_opponent:
            tank2_area = {(x, y) for x in range(self.tanks[1].position[0]-1, self.tanks[1].position[0]+2)
                                for y in range(self.tanks[1].position[1]-1, self.tanks[1].position[1]+2)}
        
        protected_area = tank1_area.union(tank2_area)
        
        while len(obstacles) < num_obstacles:
            x = random.randint(0, self.MAP_SIZE-1)
            y = random.randint(0, self.MAP_SIZE-1)
            if (x, y) not in protected_area:
                obstacles.add((x, y))
                self.current_map.add_obstacle(x, y)  # 同时更新地图中的障碍物
        return obstacles
    
    def handle_input(self):        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and not self.game_over:
                # 玩家1控制 (如果没有第二AI)
                if not self.second_ai_type:
                    if event.key == pygame.K_SPACE:  # 发射子弹
                        tank1 = self.tanks[0]
                        self.bullets.append([tank1.position[0], tank1.position[1], tank1.direction, 1])
                    elif event.key == pygame.K_a:  # 玩家1逆时针旋转
                        self.tanks[0].update_direction((self.tanks[0].direction - 1) % 4)
                    elif event.key == pygame.K_d:  # 玩家1顺时针旋转
                        self.tanks[0].update_direction((self.tanks[0].direction + 1) % 4)
                    elif event.key == pygame.K_w:  # 向当前方向前进
                        tank1 = self.tanks[0]
                        new_pos = tank1.position.copy()
                        if tank1.direction == 0:  # 上
                            new_pos[1] = max(0, new_pos[1] - 1)
                        elif tank1.direction == 1:  # 右
                            new_pos[0] = min(self.MAP_SIZE - 1, new_pos[0] + 1)
                        elif tank1.direction == 2:  # 下
                            new_pos[1] = min(self.MAP_SIZE - 1, new_pos[1] + 1)
                        elif tank1.direction == 3:  # 左
                            new_pos[0] = max(0, new_pos[0] - 1)
                        if (new_pos[0], new_pos[1]) not in self.current_map.obstacles:
                            tank1.update_position(new_pos)
                    elif event.key == pygame.K_s:  # 向当前方向后退
                        tank1 = self.tanks[0]
                        new_pos = tank1.position.copy()
                        if tank1.direction == 0:  # 上
                            new_pos[1] = min(self.MAP_SIZE - 1, new_pos[1] + 1)
                        elif tank1.direction == 1:  # 右
                            new_pos[0] = max(0, new_pos[0] - 1)
                        elif tank1.direction == 2:  # 下
                            new_pos[1] = max(0, new_pos[1] - 1)
                        elif tank1.direction == 3:  # 左
                            new_pos[0] = min(self.MAP_SIZE - 1, new_pos[0] + 1)
                        if (new_pos[0], new_pos[1]) not in self.current_map.obstacles:
                            tank1.update_position(new_pos)

                # 玩家2控制 (如果没有AI对手)
                if not self.ai_opponent:
                    if event.key == pygame.K_RETURN:  # 发射子弹
                        tank2 = self.tanks[1]
                        self.bullets.append([tank2.position[0], tank2.position[1], tank2.direction, 2])
                    elif event.key == pygame.K_LEFT:  # 玩家2逆时针旋转
                        self.tanks[1].update_direction((self.tanks[1].direction - 1) % 4)
                    elif event.key == pygame.K_RIGHT:  # 玩家2顺时针旋转
                        self.tanks[1].update_direction((self.tanks[1].direction + 1) % 4)
                    elif event.key == pygame.K_UP:  # 向当前方向前进
                        tank2 = self.tanks[1]
                        new_pos = tank2.position.copy()
                        if tank2.direction == 0:  # 上
                            new_pos[1] = max(0, new_pos[1] - 1)
                        elif tank2.direction == 1:  # 右
                            new_pos[0] = min(self.MAP_SIZE - 1, new_pos[0] + 1)
                        elif tank2.direction == 2:  # 下
                            new_pos[1] = min(self.MAP_SIZE - 1, new_pos[1] + 1)
                        elif tank2.direction == 3:  # 左
                            new_pos[0] = max(0, new_pos[0] - 1)
                        if (new_pos[0], new_pos[1]) not in self.current_map.obstacles:
                            tank2.update_position(new_pos)
                    elif event.key == pygame.K_DOWN:  # 向当前方向后退
                        tank2 = self.tanks[1]
                        new_pos = tank2.position.copy()
                        if tank2.direction == 0:  # 上
                            new_pos[1] = min(self.MAP_SIZE - 1, new_pos[1] + 1)
                        elif tank2.direction == 1:  # 右
                            new_pos[0] = max(0, new_pos[0] - 1)
                        elif tank2.direction == 2:  # 下
                            new_pos[1] = max(0, new_pos[1] - 1)
                        elif tank2.direction == 3:  # 左
                            new_pos[0] = min(self.MAP_SIZE - 1, new_pos[0] + 1)
                        if (new_pos[0], new_pos[1]) not in self.current_map.obstacles:
                            tank2.update_position(new_pos)

        # 更新AI控制的坦克
        if not self.game_over:
            if self.ai_opponent or self.second_ai_type:
                self.ai_interface.update_ai_controlled_tanks()
    
    def update_bullets(self):
        new_bullets = []
        for bullet in self.bullets:
            # 保存子弹的原始位置
            old_x, old_y = bullet[0], bullet[1]
            
            # 根据方向更新子弹位置
            if bullet[2] == 0:  # 上
                bullet[1] -= 1
            elif bullet[2] == 1:  # 右
                bullet[0] += 1
            elif bullet[2] == 2:  # 下
                bullet[1] += 1
            elif bullet[2] == 3:  # 左
                bullet[0] -= 1
            
            # 计算子弹移动路径上的所有格子
            path_points = []
            if old_x != bullet[0]:  # 水平移动
                start, end = min(old_x, bullet[0]), max(old_x, bullet[0]) + 1
                path_points = [(x, bullet[1]) for x in range(start, end)]
            elif old_y != bullet[1]:  # 垂直移动
                start, end = min(old_y, bullet[1]), max(old_y, bullet[1]) + 1
                path_points = [(bullet[0], y) for y in range(start, end)]
            
            # 检查路径上是否有障碍物
            hit_obstacle = False
            for x, y in path_points:
                if (x, y) in self.current_map.obstacles:  # 使用地图的障碍物集合
                    self.current_map.remove_obstacle(x, y)  # 使用地图类的方法移除障碍物
                    hit_obstacle = True
                    break
            
            if hit_obstacle:
                continue
                
            # 检查路径上是否击中坦克
            if bullet[3] == 1:  # 玩家1的子弹
                if self.ai_opponent:
                    for x, y in path_points:
                        if x == self.tanks[1].position[0] and y == self.tanks[1].position[1]:
                            self.game_over = True
                            self.winner = 1
                            break
                    if self.game_over:
                        continue
            else:  # 玩家2或AI的子弹
                for x, y in path_points:
                    if x == self.tanks[0].position[0] and y == self.tanks[0].position[1]:
                        self.game_over = True
                        self.winner = 2
                        break
                if self.game_over:
                    continue
            
            # 检查是否在地图范围内
            if 0 <= bullet[0] < self.MAP_SIZE and 0 <= bullet[1] < self.MAP_SIZE:
                new_bullets.append(bullet)
        
        self.bullets = new_bullets
    
    def draw_tank(self, tank, color):
        # 绘制坦克主体
        tank_rect = pygame.Rect(
            tank.position[0] * self.GRID_SIZE,
            tank.position[1] * self.GRID_SIZE,
            self.GRID_SIZE,
            self.GRID_SIZE
        )
        pygame.draw.rect(self.screen, color, tank_rect)
        
        # 绘制炮管
        center_x = tank.position[0] * self.GRID_SIZE + self.GRID_SIZE // 2
        center_y = tank.position[1] * self.GRID_SIZE + self.GRID_SIZE // 2
        if tank.direction == 0:  # 上
            pygame.draw.line(self.screen, color, (center_x, center_y), (center_x, center_y - self.GRID_SIZE))
        elif tank.direction == 1:  # 右
            pygame.draw.line(self.screen, color, (center_x, center_y), (center_x + self.GRID_SIZE, center_y))
        elif tank.direction == 2:  # 下
            pygame.draw.line(self.screen, color, (center_x, center_y), (center_x, center_y + self.GRID_SIZE))
        elif tank.direction == 3:  # 左
            pygame.draw.line(self.screen, color, (center_x, center_y), (center_x - self.GRID_SIZE, center_y))
    
    def draw(self):
        self.screen.fill(self.BLACK)
        
        # 绘制网格
        for x in range(0, self.SCREEN_SIZE, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.GRAY, (x, 0), (x, self.SCREEN_SIZE))
        for y in range(0, self.SCREEN_SIZE, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.GRAY, (0, y), (self.SCREEN_SIZE, y))
        
        # 绘制障碍物
        for obs in self.current_map.obstacles:
            obs_rect = pygame.Rect(
                obs[0] * self.GRID_SIZE,
                obs[1] * self.GRID_SIZE,
                self.GRID_SIZE,
                self.GRID_SIZE
            )
            pygame.draw.rect(self.screen, self.WHITE, obs_rect)
        
        # 绘制子弹
        for bullet in self.bullets:
            bullet_rect = pygame.Rect(
                bullet[0] * self.GRID_SIZE + self.GRID_SIZE // 3,
                bullet[1] * self.GRID_SIZE + self.GRID_SIZE // 3,
                self.GRID_SIZE // 3,
                self.GRID_SIZE // 3
            )
            bullet_color = self.RED if bullet[3] == 1 else self.BLUE
            pygame.draw.rect(self.screen, bullet_color, bullet_rect)
        
        # 绘制坦克
        self.draw_tank(self.tanks[0], self.RED)
        self.draw_tank(self.tanks[1], self.BLUE)
        
        if self.game_over:
            # 使用系统默认字体来显示中文
            font = pygame.font.SysFont(None, 30)  # 如果没有中文字体，使用默认字体
                
            if self.winner == 1:
                text = font.render('Player 1 Win!', True, self.RED)
            elif self.winner == 2:
                text = font.render('Player 2 Win!', True, self.BLUE)
            else:
                text = font.render('Game Over', True, self.WHITE)
            text_rect = text.get_rect(center=(self.SCREEN_SIZE/2, self.SCREEN_SIZE/2))
            self.screen.blit(text, text_rect)
        
        pygame.display.flip()
    
    def update_ai(self):
        if not self.ai_opponent:
            return
        
        # 简单的AI逻辑
        # 1. 朝向玩家
        dx = self.tanks[0].position[0] - self.tanks[1].position[0]
        dy = self.tanks[0].position[1] - self.tanks[1].position[1]
        
        # 决定移动方向
        new_pos = self.tanks[1].position.copy()
        
        # 优先水平移动
        if abs(dx) > abs(dy):
            if dx > 0:
                new_direction = 1  # 右
                new_pos[0] = min(self.MAP_SIZE - 1, new_pos[0] + 1)
            else:
                new_direction = 3  # 左
                new_pos[0] = max(0, new_pos[0] - 1)
        else:
            if dy > 0:
                new_direction = 2  # 下
                new_pos[1] = min(self.MAP_SIZE - 1, new_pos[1] + 1)
            else:
                new_direction = 0  # 上
                new_pos[1] = max(0, new_pos[1] - 1)
        
        # 检查碰撞
        if (new_pos[0], new_pos[1]) not in self.current_map.obstacles:
            self.tanks[1].update_position(new_pos)
            self.tanks[1].update_direction(new_direction)
        
        # 发射子弹
        if random.random() < 0.3:  # 30%的概率发射子弹
            self.bullets.append([self.tanks[1].position[0], self.tanks[1].position[1], self.tanks[1].direction, 2])
    
    def check_game_over(self):
        # 游戏结束检查已移至update_bullets方法中
        pass
    
    def run(self):
        """游戏主循环"""
        clock = pygame.time.Clock()
        frame_count = 0
        
        while self.running:
            frame_count += 1
            
            # 处理输入
            self.handle_input()
            
            if not self.game_over:
                # 更新子弹
                self.update_bullets()
                
                # AI状态输出（每秒一次）
                if self.ai_opponent and frame_count % 10 == 0:
                    print("\nAI状态更新:")
                    print(f"坦克1 - 位置: {self.tanks[0].position}, 朝向: {self.tanks[0].direction}")
                    print(f"坦克2 - 位置: {self.tanks[1].position}, 朝向: {self.tanks[1].direction}")
            
            # 渲染游戏
            self.draw()
            
            # 控制帧率
            clock.tick(10)  # 限制帧率为10 FPS
        
        pygame.quit()

def main(ai_opponent=False, ai_type='dqn', second_ai_type=None):
    game = SimplifiedGame(ai_opponent=ai_opponent, ai_type=ai_type, second_ai_type=second_ai_type)
    game.run()
