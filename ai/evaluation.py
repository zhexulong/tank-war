import numpy as np
from typing import Dict, Tuple


def evaluate_state(obs_state, player_id, w1, w2, w3, w4):
    """Your scoring function with risk evaluation"""
    R_ally, R_enemy = risk_evaluation(obs_state, player_id)
    
    # Calculate rewards and punishments
    # Win/lose reward based on game state
    R_win = 100 if obs_state['game_over'] and obs_state['winner'] == player_id else 0.0
    R_lose = 100 if obs_state['game_over'] and obs_state['winner'] != player_id and obs_state['winner'] is not None else 0.0

    # Distance penalty
    P_distance = distance_penalty(obs_state, player_id)
    
    # Strategic rewards
    S_path, S_firing = evaluate_strategic_position(obs_state, player_id)
    
    #print("R_ally,R_enemy,R_win,R_lose,P_distance,S_path,S_firing: ",R_ally,R_enemy,R_win,R_lose,P_distance,S_path,S_firing)
    return -w1*R_ally + w2*(R_enemy + S_firing) + w3*(R_win - R_lose + S_path) - w4*P_distance


def distance_penalty(obs_state: Dict, player_id: int) -> float:
    """Calculate distance penalty that increases exponentially with distance
    
    The penalty is calculated as: (dist / threshold)^2
    where threshold is half the map diagonal length.
    This means:
    - At half the map diagonal: penalty = 1.0
    - At map diagonal: penalty = 4.0
    - Close distance: penalty approaches 0
    """
    # Find our tank and enemy tank
    our_tank = None
    enemy_tank = None
    for tank in obs_state['tanks']:
        if tank['player_id'] == player_id:
            our_tank = tank
        else:
            enemy_tank = tank
    
    if not our_tank or not enemy_tank:
        return 0.0
    
    # Calculate distance
    our_pos = np.array(our_tank['position'])
    enemy_pos = np.array(enemy_tank['position'])
    dist = np.linalg.norm(our_pos - enemy_pos)
    
    # Calculate map diagonal length
    map_width = len(obs_state['map'][0])
    map_height = len(obs_state['map'])
    diagonal = np.sqrt(map_width**2 + map_height**2)
    
    # Use half diagonal as threshold for normal penalty
    threshold = diagonal / 2
    
    # Calculate quadratic penalty
    penalty = (dist / threshold) ** 2
    
    return penalty  # Can exceed 1.0 for very large distances

def is_aligned(pos1: np.ndarray, pos2: np.ndarray, tolerance: float = 0.5) -> Tuple[bool, int]:
    """Check if two positions are aligned horizontally or vertically
    Returns: (is_aligned, direction)
    direction: 0=vertical, 1=horizontal, -1=not aligned
    """
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])
    
    if dx < tolerance:  # Same column
        return True, 0
    elif dy < tolerance:  # Same row
        return True, 1
    return False, -1

def calculate_directional_threat(tank_pos: np.ndarray, tank_dir: int, target_pos: np.ndarray) -> float:
    """Calculate how threatening a tank's direction is based on its target
    Returns a value between 0 and 1, where 1 means the tank is pointing directly at the target
    """
    # Calculate angle to target
    to_target = target_pos - tank_pos
    target_angle = np.arctan2(to_target[1], to_target[0])
    if target_angle < 0:
        target_angle += 2 * np.pi
    
    # Convert tank direction to radians
    tank_angle = tank_dir * np.pi / 2  # Convert 0-3 to radians
    
    # Calculate angle difference
    angle_diff = abs(target_angle - tank_angle)
    angle_diff = min(angle_diff, 2 * np.pi - angle_diff)  # Take smaller angle
    
    # Convert to threat level (0 to 1)
    # Full threat when pointing directly (angle_diff = 0)
    # Zero threat when pointing opposite (angle_diff = pi)
    threat = max(0, 1 - angle_diff / np.pi)
    return threat

def risk_evaluation(obs_state: Dict, player_id: int) -> Tuple[float, float]:
    """
    Evaluate risk for both ally (player) and enemy tanks
    Returns: (R_ally, R_enemy) risk scores between [0,1]
    """
    def calculate_risk(tank: Dict) -> float:
        """Calculate risk for a tank"""
        if not tank:
            return 0.0
            
        total_risk = 0.0
        tank_pos = np.array(tank['position'])
        
        # 1. Enemy proximity and directional risk
        enemy = None
        for t in obs_state['tanks']:
            if t['player_id'] != tank['player_id']:
                enemy = t
                break
        
        if enemy:
            enemy_pos = np.array(enemy['position'])
            dist = np.linalg.norm(tank_pos - enemy_pos)
            
            # Check if tanks are aligned
            is_align, align_dir = is_aligned(tank_pos, enemy_pos)
            
            # Base proximity risk
            proximity_risk = np.exp(-dist)  # Risk decays exponentially with distance
            
            # Additional risk if tanks are aligned
            if is_align:
                # Calculate how much enemy is pointing at us
                enemy_threat = calculate_directional_threat(enemy_pos, enemy['direction'], tank_pos)
                # Higher risk if enemy is pointing at us and we're aligned
                proximity_risk *= (1.0 + enemy_threat)
            
            total_risk += proximity_risk * 0.4
        
        # 2. Bullet trajectory risk
        bullet_risk = 0.0
        for bullet in obs_state['bullets']:
            bullet_pos = np.array(bullet[:2])
            bullet_dir = bullet[2]
            bullet_owner = bullet[3]
            
            if bullet_owner != tank['player_id']:  # Enemy bullet
                bullet_to_tank = tank_pos - bullet_pos
                dist = np.linalg.norm(bullet_to_tank)
                
                # Check alignment with different tolerances
                is_direct_hit, align_dir = is_aligned(bullet_pos, tank_pos, tolerance=0.5)
                is_near_hit, _ = is_aligned(bullet_pos, tank_pos, tolerance=1.5)
                
                # Verify bullet is moving towards tank
                is_approaching = False
                if bullet_dir == 0 and bullet_pos[1] > tank_pos[1]:  # Up
                    is_approaching = True
                elif bullet_dir == 1 and bullet_pos[0] < tank_pos[0]:  # Right
                    is_approaching = True
                elif bullet_dir == 2 and bullet_pos[1] < tank_pos[1]:  # Down
                    is_approaching = True
                elif bullet_dir == 3 and bullet_pos[0] > tank_pos[0]:  # Left
                    is_approaching = True
                
                if is_approaching:
                    if is_direct_hit:
                        # Direct hit potential: highest risk
                        bullet_risk += 2.0 * np.exp(-dist/3)  # Slower decay for direct hits
                    elif is_near_hit:
                        # Near hit: medium risk
                        bullet_risk += 1.0 * np.exp(-dist/2)  # Medium decay for near hits
                    else:
                        # Distant bullet: low risk
                        bullet_risk += 0.2 * np.exp(-dist)  # Fast decay for distant bullets
        
        total_risk += bullet_risk  # No need to cap since we use exponential decay
        
        # 3. Obstacle coverage risk (less cover = more risk)
        cover_score = 0
        pos_x, pos_y = int(tank_pos[0]), int(tank_pos[1])
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = pos_x + dx, pos_y + dy
                if 0 <= nx < len(obs_state['map'][0]) and 0 <= ny < len(obs_state['map']):
                    if (nx, ny) in obs_state['obstacles']:
                        cover_score += 1
        
        coverage_risk = 1.0 - (cover_score / 8)  # Normalized [0,1]
        total_risk += coverage_risk * 0.2
        
        return min(total_risk, 1.0)  # Cap final risk at 1.0
    
    # Find our tank and enemy tank
    our_tank = None
    enemy_tank = None
    for tank in obs_state['tanks']:
        if tank['player_id'] == player_id:
            our_tank = tank
        else:
            enemy_tank = tank
    
    if not our_tank or not enemy_tank:
        return 0.0, 0.0
    
    # Calculate risks
    R_ally = calculate_risk(our_tank)
    R_enemy = calculate_risk(enemy_tank)
    
    return R_ally, R_enemy

def evaluate_strategic_position(obs_state: Dict, player_id: int) -> Tuple[float, float]:
    """Evaluate strategic advantages of position and firing opportunities
    Returns: (path_score, firing_score)
    """
    # Find our tank and enemy tank
    our_tank = None
    enemy_tank = None
    for tank in obs_state['tanks']:
        if tank['player_id'] == player_id:
            our_tank = tank
        else:
            enemy_tank = tank
    
    if not our_tank or not enemy_tank:
        return 0.0, 0.0
    
    our_pos = np.array(our_tank['position'])
    enemy_pos = np.array(enemy_tank['position'])
    
    # Calculate path clearance score
    path_score = evaluate_path_clearance(obs_state, our_pos, enemy_pos)
    
    # Calculate firing opportunity score
    firing_score = evaluate_firing_opportunity(obs_state, our_tank, enemy_tank)
    
    return path_score, firing_score

def evaluate_path_clearance(obs_state: Dict, our_pos: np.ndarray, enemy_pos: np.ndarray) -> float:
    """Evaluate how clear the path is towards the enemy"""
    # Get direction to enemy
    to_enemy = enemy_pos - our_pos
    distance = np.linalg.norm(to_enemy)
    direction = np.arctan2(to_enemy[1], to_enemy[0])
    
    # Check for obstacles in the path
    obstacles_in_path = 0
    step = 1.0
    current_pos = our_pos.copy()
    
    while np.linalg.norm(current_pos - our_pos) < distance:
        current_pos[0] += step * np.cos(direction)
        current_pos[1] += step * np.sin(direction)
        grid_pos = tuple(np.round(current_pos).astype(int))
        
        if grid_pos in obs_state['obstacles']:
            obstacles_in_path += 1
    
    # Calculate path score (higher is better)
    # Score decreases with number of obstacles but increases with distance
    # This encourages clearing paths when enemy is far
    path_score = 1.0 / (1.0 + obstacles_in_path) * np.log1p(distance)
    
    return path_score

def evaluate_firing_opportunity(obs_state: Dict, our_tank: Dict, enemy_tank: Dict) -> float:
    """Evaluate firing opportunities considering bullet advantage and position"""
    our_pos = np.array(our_tank['position'])
    enemy_pos = np.array(enemy_tank['position'])
    
    # Count bullets in the line of fire
    our_bullets = 0
    enemy_bullets = 0
    is_align, align_dir = is_aligned(our_pos, enemy_pos)
    
    if is_align:
        for bullet in obs_state['bullets']:
            bullet_pos = np.array(bullet[:2])
            # Check if bullet is in the same line
            if is_aligned(bullet_pos, our_pos)[0]:
                if bullet[3] == our_tank['player_id']:
                    our_bullets += 1
                else:
                    enemy_bullets += 1
    
    # Calculate distance-based firing score
    distance = np.linalg.norm(enemy_pos - our_pos)
    base_firing_score = 1.0 / (1.0 + distance)  # Higher score for closer enemies
    
    # Adjust score based on bullet advantage and alignment
    if is_align:
        if enemy_bullets == 0:
            # No enemy bullets - great opportunity to fire
            base_firing_score *= 2.0
        elif our_bullets > enemy_bullets:
            # We have bullet advantage - good to fire
            base_firing_score *= 1.5
        elif our_bullets < enemy_bullets:
            # Enemy has bullet advantage - reduce score
            base_firing_score *= 0.5
    else:
        # Not aligned - reduce score but don't eliminate it
        # Still want to encourage firing to clear paths
        if distance < 3 :
            # Encourage getting into position at close range when safe
            base_firing_score *= 0.8
        else:
            base_firing_score *= 0.3
    
    # Check if we're pointing towards enemy
    direction_score = calculate_directional_threat(our_pos, our_tank['direction'], enemy_pos)
    
    # Add movement prediction for close combat
    if distance < 4:
        # Predict if enemy is moving towards us
        enemy_direction = enemy_tank['direction']
 
           
    # Combine scores
    firing_score = base_firing_score * (1.0 + direction_score)
    
    return firing_score