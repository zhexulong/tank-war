# Import required modules
import pygame
import random
from ai.simplified_interface import SimplifiedAIInterface
from game.core import GameCore

# Terrain type constants
EMPTY = 0
BRICK = 1
STEEL = 2
WATER = 3
GRASS = 4

class SimplifiedMap:
    """Simplified map class"""
    def __init__(self, width, height, tile_size):
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.grid = [[EMPTY for _ in range(width)] for _ in range(height)]
        self.obstacles = set()  # Store obstacle positions
    
    def get_tile(self, x, y):
        """Get terrain type at specified position"""
        if 0 <= x < self.width and 0 <= y < self.height:
            if (x, y) in self.obstacles:
                return BRICK
            return EMPTY
        return STEEL  # Map boundaries are treated as steel walls
    
    def add_obstacle(self, x, y):
        """Add obstacle"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.obstacles.add((x, y))
    
    def remove_obstacle(self, x, y):
        """Remove obstacle"""
        if (x, y) in self.obstacles:
            self.obstacles.remove((x, y))
    
    def is_obstacle(self, x, y):
        """Check if position has obstacle"""
        return (x, y) in self.obstacles

class SimplifiedTank:
    """Simplified tank class"""
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
    def __init__(self, ai_opponent=False, ai_type='dqn', second_ai_type=None, render_mode=None, agent_path=None):
        # Game constants
        self.GRID_SIZE = 50  # Pixel size of each grid cell
        self.MAP_SIZE = 16   # Map size (in grid cells)
        self.SCREEN_SIZE = self.GRID_SIZE * self.MAP_SIZE  # Screen size
        
        # AI model path
        self.agent_path = agent_path
        
        # Render mode
        self.render_mode = render_mode
        
        # Pygame initialization and screen setup (only when rendering)
        self.screen = None
        if self.render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_SIZE, self.SCREEN_SIZE))
            pygame.display.set_caption('Tank Battle')
        
        # Color definitions
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (128, 128, 128)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        # Additional colors
        self.DARK_GRAY = (40, 40, 40)
        self.LIGHT_GRAY = (200, 200, 200)
        self.BRICK_COLOR = (139, 69, 19)  # Brick wall color
        self.TANK_BLUE = (30, 144, 255)   # Player tank color
        self.TANK_RED = (220, 20, 60)     # AI tank color
        self.BULLET_COLOR = (255, 215, 0)  # Bullet color
        self.GRID_COLOR = (50, 50, 50)    # Grid line color
        # Tank detail colors
        self.TANK_BLUE_DARK = (0, 100, 200)   # Player tank dark color
        self.TANK_RED_DARK = (180, 0, 0)      # AI tank dark color
        self.TANK_HIGHLIGHT = (255, 255, 255, 128)  # Tank highlight
        
        # Create map
        self.current_map = SimplifiedMap(self.MAP_SIZE, self.MAP_SIZE, self.GRID_SIZE)
        
        # Game state
        self.running = True
        self.game_over = False
        self.winner = None
        
        # Game elements
        self.tanks = []
        self.bullets = []
        
        # AI settings
        self.ai_opponent = ai_opponent
        self.ai_type = ai_type
        self.second_ai_type = second_ai_type
        self.ai_interface = None
        
        # Initialize game
        self.init_game()
        
        # Initialize AI if enabled
        if self.ai_opponent:
            self.setup_ai()
    
    def init_game(self):
        """Initialize game state"""
        # Reset tank positions
        self.tanks = [
            SimplifiedTank(2, 2, 0, 1),  # Player 1 tank
            SimplifiedTank(self.MAP_SIZE - 3, self.MAP_SIZE - 3, 2, 2)  # Player 2 tank
        ]
        
        # Clear bullets
        self.bullets = []
        
        # Reset game state
        self.game_over = False
        self.winner = None
        
        # Generate obstacles
        self.current_map.obstacles = self.generate_obstacles()
    
    def setup_ai(self):
        """Set up AI system"""
        game_core = GameCore(self)
        self.ai_interface = SimplifiedAIInterface(self, game_core)
        
        print(f"Initializing AI - Type: {self.ai_type}")
        # Initialize AI (Player 2)
        self.ai_interface.load_agent(2, self.agent_path, self.ai_type)
        self.tanks[1].is_player = False
        
        # If it's AI vs AI mode
        if self.second_ai_type:
            print(f"Initializing second AI - Type: {self.second_ai_type}")
            self.ai_interface.load_agent(1, self.agent_path, self.second_ai_type)
            self.tanks[0].is_player = False  # Set player 1 tank to AI control

    def get_player_tank(self, player_id):
        """Get specified player's tank"""
        for tank in self.tanks:
            if tank.player_id == player_id:
                return tank
        return None

    def get_game_state_for_rl(self):
        """Get game state for reinforcement learning"""
        # Regenerate map grid: 0 represents empty land, 1 represents obstacle
        map_grid = [[1 if self.current_map.is_obstacle(x, y) else 0
                     for x in range(self.MAP_SIZE)]
                    for y in range(self.MAP_SIZE)]
        return {
            'map': map_grid,
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
        """Get shape of game state space"""
        return {
            'map': (self.MAP_SIZE, self.MAP_SIZE),  # Map grid
            'tanks': (2, 4),  # Two tanks, each tank [x,y,direction,player_id]
            'bullets': (10, 3),  # Up to 10 bullets, each bullet [x,y,direction]
        }
    
    def generate_obstacles(self, num_obstacles=50):
        obstacles = set()
        # Protect initial area of two tanks
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
                self.current_map.add_obstacle(x, y)  # Also update obstacles in map
        return obstacles
    
    def handle_input(self):        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r and self.game_over:  # Add restart functionality
                    self.restart_game()
                elif not self.game_over:
                    # Player 1 control (if no second AI)
                    if not self.second_ai_type:
                        if event.key == pygame.K_SPACE:  # Fire bullet
                            tank1 = self.tanks[0]
                            self.bullets.append([tank1.position[0], tank1.position[1], tank1.direction, 1])
                        elif event.key == pygame.K_a:  # Player 1 counterclockwise rotation
                            self.tanks[0].update_direction((self.tanks[0].direction - 1) % 4)
                        elif event.key == pygame.K_d:  # Player 1 clockwise rotation
                            self.tanks[0].update_direction((self.tanks[0].direction + 1) % 4)
                        elif event.key == pygame.K_w:  # Move forward in current direction
                            tank1 = self.tanks[0]
                            new_pos = tank1.position.copy()
                            if tank1.direction == 0:  # Up
                                new_pos[1] = max(0, new_pos[1] - 1)
                            elif tank1.direction == 1:  # Right
                                new_pos[0] = min(self.MAP_SIZE - 1, new_pos[0] + 1)
                            elif tank1.direction == 2:  # Down
                                new_pos[1] = min(self.MAP_SIZE - 1, new_pos[1] + 1)
                            elif tank1.direction == 3:  # Left
                                new_pos[0] = max(0, new_pos[0] - 1)
                            if (new_pos[0], new_pos[1]) not in self.current_map.obstacles:
                                tank1.update_position(new_pos)
                        elif event.key == pygame.K_s:  # Move backward in current direction
                            tank1 = self.tanks[0]
                            new_pos = tank1.position.copy()
                            if tank1.direction == 0:  # Up
                                new_pos[1] = min(self.MAP_SIZE - 1, new_pos[1] + 1)
                            elif tank1.direction == 1:  # Right
                                new_pos[0] = max(0, new_pos[0] - 1)
                            elif tank1.direction == 2:  # Down
                                new_pos[1] = max(0, new_pos[1] - 1)
                            elif tank1.direction == 3:  # Left
                                new_pos[0] = min(self.MAP_SIZE - 1, new_pos[0] + 1)
                            if (new_pos[0], new_pos[1]) not in self.current_map.obstacles:
                                tank1.update_position(new_pos)

                # Player 2 control (if no AI opponent)
                if not self.ai_opponent:
                    if event.key == pygame.K_RETURN:  # Fire bullet
                        tank2 = self.tanks[1]
                        self.bullets.append([tank2.position[0], tank2.position[1], tank2.direction, 2])
                    elif event.key == pygame.K_LEFT:  # Player 2 counterclockwise rotation
                        self.tanks[1].update_direction((self.tanks[1].direction - 1) % 4)
                    elif event.key == pygame.K_RIGHT:  # Player 2 clockwise rotation
                        self.tanks[1].update_direction((self.tanks[1].direction + 1) % 4)
                    elif event.key == pygame.K_UP:  # Move forward in current direction
                        tank2 = self.tanks[1]
                        new_pos = tank2.position.copy()
                        if tank2.direction == 0:  # Up
                            new_pos[1] = max(0, new_pos[1] - 1)
                        elif tank2.direction == 1:  # Right
                            new_pos[0] = min(self.MAP_SIZE - 1, new_pos[0] + 1)
                        elif tank2.direction == 2:  # Down
                            new_pos[1] = min(self.MAP_SIZE - 1, new_pos[1] + 1)
                        elif tank2.direction == 3:  # Left
                            new_pos[0] = max(0, new_pos[0] - 1)
                        if (new_pos[0], new_pos[1]) not in self.current_map.obstacles:
                            tank2.update_position(new_pos)
                    elif event.key == pygame.K_DOWN:  # Move backward in current direction
                        tank2 = self.tanks[1]
                        new_pos = tank2.position.copy()
                        if tank2.direction == 0:  # Up
                            new_pos[1] = min(self.MAP_SIZE - 1, new_pos[1] + 1)
                        elif tank2.direction == 1:  # Right
                            new_pos[0] = max(0, new_pos[0] - 1)
                        elif tank2.direction == 2:  # Down
                            new_pos[1] = max(0, new_pos[1] - 1)
                        elif tank2.direction == 3:  # Left
                            new_pos[0] = min(self.MAP_SIZE - 1, new_pos[0] + 1)
                        if (new_pos[0], new_pos[1]) not in self.current_map.obstacles:
                            tank2.update_position(new_pos)

        # Update AI-controlled tanks
        if not self.game_over:
            if self.ai_opponent or self.second_ai_type:
                self.ai_interface.update_ai_controlled_tanks()
    
    def update_bullets(self):
        new_bullets = []
        for bullet in self.bullets:
            # Save bullet's original position
            old_x, old_y = bullet[0], bullet[1]
            
            # Update bullet position based on direction
            if bullet[2] == 0:  # Up
                bullet[1] -= 1
            elif bullet[2] == 1:  # Right
                bullet[0] += 1
            elif bullet[2] == 2:  # Down
                bullet[1] += 1
            elif bullet[2] == 3:  # Left
                bullet[0] -= 1
            
            # Calculate all grid cells in bullet's path
            path_points = []
            if old_x != bullet[0]:  # Horizontal movement
                start, end = min(old_x, bullet[0]), max(old_x, bullet[0]) + 1
                path_points = [(x, bullet[1]) for x in range(start, end)]
            elif old_y != bullet[1]:  # Vertical movement
                start, end = min(old_y, bullet[1]), max(old_y, bullet[1]) + 1
                path_points = [(bullet[0], y) for y in range(start, end)]
            
            # Check for obstacles in path
            hit_obstacle = False
            for x, y in path_points:
                if (x, y) in self.current_map.obstacles:
                    self.current_map.remove_obstacle(x, y)
                    hit_obstacle = True
                    break
            
            if hit_obstacle:
                continue
                
            # Check for tank hits in path
            if bullet[3] == 1:  # Player 1's bullet
                if self.ai_opponent:
                    for x, y in path_points:
                        if x == self.tanks[1].position[0] and y == self.tanks[1].position[1]:
                            self.game_over = True
                            self.winner = 1
                            print("Player 1 wins")  # Add print statement
                            break
                    if self.game_over:
                        continue
            else:  # Player 2 or AI's bullet
                for x, y in path_points:
                    if x == self.tanks[0].position[0] and y == self.tanks[0].position[1]:
                        self.game_over = True
                        self.winner = 2
                        print("Player 2 wins")  # Add print statement
                        break
                if self.game_over:
                    continue
            
            # Check if within map boundaries
            if 0 <= bullet[0] < self.MAP_SIZE and 0 <= bullet[1] < self.MAP_SIZE:
                new_bullets.append(bullet)
        
        self.bullets = new_bullets
    
    def draw_tank(self, tank, color, dark_color):
        if self.render_mode != 'human' or self.screen is None:
            return  # Return if no rendering needed or screen not initialized
            
        # Calculate tank position and size
        x = tank.position[0] * self.GRID_SIZE
        y = tank.position[1] * self.GRID_SIZE
        size = self.GRID_SIZE
        
        # Calculate center point
        center_x = x + size // 2
        center_y = y + size // 2
        radius = size // 2 - 4  # Slightly smaller to leave margin
        
        # Draw tank body (circle)
        # Outer circle (dark color)
        pygame.draw.circle(self.screen, dark_color, (center_x, center_y), radius + 2)
        # Main body (main color)
        pygame.draw.circle(self.screen, color, (center_x, center_y), radius)
        
        # Draw tank tracks
        track_width = 4
        track_length = size - 8
        # Left track
        pygame.draw.rect(self.screen, dark_color, 
                        (x + 2, y + size//2 - track_width//2, track_length, track_width))
        # Right track
        pygame.draw.rect(self.screen, dark_color, 
                        (x + 2, y + size//2 + track_width//2, track_length, track_width))
        
        # Draw turret (circle)
        turret_radius = radius * 0.7
        pygame.draw.circle(self.screen, dark_color, (center_x, center_y), turret_radius + 1)
        pygame.draw.circle(self.screen, color, (center_x, center_y), turret_radius)
        
        # Draw barrel
        barrel_length = size * 0.8
        barrel_width = 6
        if tank.direction == 0:  # Up
            pygame.draw.rect(self.screen, dark_color, 
                           (center_x - barrel_width//2, center_y - barrel_length, 
                            barrel_width, barrel_length))
        elif tank.direction == 1:  # Right
            pygame.draw.rect(self.screen, dark_color, 
                           (center_x, center_y - barrel_width//2, 
                            barrel_length, barrel_width))
        elif tank.direction == 2:  # Down
            pygame.draw.rect(self.screen, dark_color, 
                           (center_x - barrel_width//2, center_y, 
                            barrel_width, barrel_length))
        elif tank.direction == 3:  # Left
            pygame.draw.rect(self.screen, dark_color, 
                           (center_x - barrel_length, center_y - barrel_width//2, 
                            barrel_length, barrel_width))
        
        # Add highlight effect
        highlight_radius = radius * 0.3
        highlight_pos = (center_x - radius//3, center_y - radius//3)
        pygame.draw.circle(self.screen, self.TANK_HIGHLIGHT, highlight_pos, highlight_radius)
        
        # Add turret details
        detail_radius = turret_radius * 0.5
        pygame.draw.circle(self.screen, dark_color, (center_x, center_y), detail_radius)
        pygame.draw.circle(self.screen, color, (center_x, center_y), detail_radius - 1)
        
        # Add barrel details
        if tank.direction == 0:  # Up
            pygame.draw.rect(self.screen, color, 
                           (center_x - barrel_width//2 + 1, center_y - barrel_length, 
                            barrel_width - 2, barrel_length))
        elif tank.direction == 1:  # Right
            pygame.draw.rect(self.screen, color, 
                           (center_x, center_y - barrel_width//2 + 1, 
                            barrel_length, barrel_width - 2))
        elif tank.direction == 2:  # Down
            pygame.draw.rect(self.screen, color, 
                           (center_x - barrel_width//2 + 1, center_y, 
                            barrel_width - 2, barrel_length))
        elif tank.direction == 3:  # Left
            pygame.draw.rect(self.screen, color, 
                           (center_x - barrel_length, center_y - barrel_width//2 + 1, 
                            barrel_length, barrel_width - 2))
    
    def draw(self):
        """Draw game state"""
        if self.render_mode != 'human' or self.screen is None:
            return  # Return if no rendering needed or screen not initialized
            
        # Fill background
        self.screen.fill(self.DARK_GRAY)
        
        # Draw grid lines
        for x in range(0, self.SCREEN_SIZE, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.GRID_COLOR, (x, 0), (x, self.SCREEN_SIZE))
        for y in range(0, self.SCREEN_SIZE, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.GRID_COLOR, (0, y), (self.SCREEN_SIZE, y))
        
        # Draw map
        for x in range(self.MAP_SIZE):
            for y in range(self.MAP_SIZE):
                if self.current_map.is_obstacle(x, y):
                    # Draw obstacle (brick wall effect)
                    obstacle_rect = pygame.Rect(
                        x * self.GRID_SIZE + 1,
                        y * self.GRID_SIZE + 1,
                        self.GRID_SIZE - 2,
                        self.GRID_SIZE - 2
                    )
                    pygame.draw.rect(self.screen, self.BRICK_COLOR, obstacle_rect)
                    # Add brick texture
                    for i in range(2):
                        for j in range(2):
                            brick = pygame.Rect(
                                x * self.GRID_SIZE + 1 + i * (self.GRID_SIZE // 2),
                                y * self.GRID_SIZE + 1 + j * (self.GRID_SIZE // 2),
                                self.GRID_SIZE // 2 - 1,
                                self.GRID_SIZE // 2 - 1
                            )
                            pygame.draw.rect(self.screen, (160, 82, 45), brick, 1)
        
        # Draw tanks
        self.draw_tank(self.tanks[0], self.TANK_BLUE, self.TANK_BLUE_DARK)
        self.draw_tank(self.tanks[1], self.TANK_RED, self.TANK_RED_DARK)
        
        # Draw bullets
        for bullet in self.bullets:
            bullet_x = bullet[0] * self.GRID_SIZE + self.GRID_SIZE // 2
            bullet_y = bullet[1] * self.GRID_SIZE + self.GRID_SIZE // 2
            # Draw glowing bullet effect
            pygame.draw.circle(self.screen, self.WHITE, (bullet_x, bullet_y), 4)
            pygame.draw.circle(self.screen, self.BULLET_COLOR, (bullet_x, bullet_y), 2)
        
        # Draw game over information
        if self.game_over:
            # Create semi-transparent overlay
            overlay = pygame.Surface((self.SCREEN_SIZE, self.SCREEN_SIZE))
            overlay.set_alpha(128)
            overlay.fill(self.BLACK)
            self.screen.blit(overlay, (0, 0))
            
            # Draw win message
            font = pygame.font.Font(None, 48)
            text = font.render(f"player {self.winner} wins!", True, self.WHITE)
            text_rect = text.get_rect(center=(self.SCREEN_SIZE // 2, self.SCREEN_SIZE // 2))
            
            # Add text shadow effect
            shadow = font.render(f"player {self.winner} wins!", True, self.DARK_GRAY)
            shadow_rect = shadow.get_rect(center=(self.SCREEN_SIZE // 2 + 2, self.SCREEN_SIZE // 2 + 2))
            self.screen.blit(shadow, shadow_rect)
            self.screen.blit(text, text_rect)
            
            # Add restart prompt
            small_font = pygame.font.Font(None, 24)
            restart_text = small_font.render("press R to restart or ESC to exit", True, self.LIGHT_GRAY)
            restart_rect = restart_text.get_rect(center=(self.SCREEN_SIZE // 2, self.SCREEN_SIZE // 2 + 40))
            self.screen.blit(restart_text, restart_rect)
        
        pygame.display.flip()
    
    def update_ai(self):
        if not self.ai_opponent:
            return
        
        # Simple AI logic
        # 1. Face the player
        dx = self.tanks[0].position[0] - self.tanks[1].position[0]
        dy = self.tanks[0].position[1] - self.tanks[1].position[1]
        
        # Decide movement direction
        new_pos = self.tanks[1].position.copy()
        
        # Prioritize horizontal movement
        if abs(dx) > abs(dy):
            if dx > 0:
                new_direction = 1  # Right
                new_pos[0] = min(self.MAP_SIZE - 1, new_pos[0] + 1)
            else:
                new_direction = 3  # Left
                new_pos[0] = max(0, new_pos[0] - 1)
        else:
            if dy > 0:
                new_direction = 2  # Down
                new_pos[1] = min(self.MAP_SIZE - 1, new_pos[1] + 1)
            else:
                new_direction = 0  # Up
                new_pos[1] = max(0, new_pos[1] - 1)
        
        # Check collision
        if (new_pos[0], new_pos[1]) not in self.current_map.obstacles:
            self.tanks[1].update_position(new_pos)
            self.tanks[1].update_direction(new_direction)
        
        # Fire bullet
        if random.random() < 0.3:  # 30% chance to fire
            self.bullets.append([self.tanks[1].position[0], self.tanks[1].position[1], self.tanks[1].direction, 2])
    
    def check_game_over(self):
        # Game over check moved to update_bullets method
        pass
    
    def close(self):
        """Close game and release resources"""
        try:
            # Only call pygame.quit() if in human render mode and pygame is initialized
            if self.render_mode == 'human' and pygame.get_init():
                pygame.quit()
        except Exception as e:
            pass  # Ignore exception if pygame is already closed
    
    def run(self):
        """Game main loop"""
        clock = None
        if self.render_mode == 'human':
            clock = pygame.time.Clock()
        frame_count = 0
        
        while self.running:
            frame_count += 1
            
            # Handle input
            if self.render_mode == 'human':
                self.handle_input()
            
            if not self.game_over:
                # Update bullets
                self.update_bullets()
                
                # Update AI-controlled tanks
                if self.ai_opponent or self.second_ai_type:
                    self.ai_interface.update_ai_controlled_tanks()
                '''
                # AI status output (once per second)
                if self.ai_opponent and frame_count % 10 == 0:
                    print("\nAI Status Update:")
                    print(f"Tank 1 - Position: {self.tanks[0].position}, Direction: {self.tanks[0].direction}")
                    print(f"Tank 2 - Position: {self.tanks[1].position}, Direction: {self.tanks[1].direction}")
                '''
            else:
                # In non-render mode, terminate the game when there's a winner
                if self.render_mode != 'human':
                    self.running = False
                    break
            
            # Render game
            self.draw()
            if self.render_mode == 'human':
                pygame.display.flip()
                clock.tick(10)  # Limit to 10 FPS
        
        self.close()

    def restart_game(self):
        """重新开始游戏"""
        # 重置游戏状态
        self.game_over = False
        self.winner = None
        self.bullets = []
        
        # 重置坦克位置
        self.tanks[0].position = [2, 2]
        self.tanks[0].direction = 0
        self.tanks[1].position = [self.MAP_SIZE - 3, self.MAP_SIZE - 3]
        self.tanks[1].direction = 2
        
        # 重新生成障碍物
        self.current_map.obstacles = self.generate_obstacles()

def main(ai_opponent=False, ai_type='dqn', second_ai_type=None, render_mode='human', agent_path=None):
    game = SimplifiedGame(ai_opponent=ai_opponent, ai_type=ai_type, second_ai_type=second_ai_type, render_mode=render_mode, agent_path=agent_path)
    game.run()
