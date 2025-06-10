from typing import Tuple, Dict, Optional, List
import numpy as np
from ai.evaluation import evaluate_state
from ai.base_agent import BaseAgent

# Constants
POSI_INFI = 2147483647
NEGA_INFI = -2147483647
DEFAULT_WEIGHTS = {
    'w1': 1,  # Ally risk
    'w2': 30,  # Enemy risk
    'w3': 1,  # Win reward
    'w4': 0.1   # Distance punishment
}



class MinimaxAgent(BaseAgent):
    """Minimax-based tank agent"""
    def __init__(self, depth: int = 3, weights: Dict = DEFAULT_WEIGHTS):
        self.depth = depth
        self.weights = weights
    
    def select_action(self, state: Dict, training: bool = False) -> int:
        """Select action using minimax
        
        Args:
            state: Simplified game state with:
                - 'map': 2D grid of 0s and 1s
                - 'tanks': List of tank dicts with position, direction, player_id
                - 'bullets': List of bullet lists [x, y, direction, owner_id]
                - 'obstacles': Set of obstacle coordinates
                - 'game_over' and 'winner' states
        Returns:
            action: 0=no_action, 1=forward, 2=rotate_left, 3=rotate_right, 4=fire
        """
        _, action = alpha_beta_tank(
            obs_state=state,
            player_id=self._get_player_id(state),
            depth=self.depth,
            alpha= NEGA_INFI,
            beta = POSI_INFI,
            is_maximizing=True,
        )
        return action or 0  # Default to no action
    
    def _get_player_id(self, state: Dict) -> int:
        """Determine player ID from state"""
        return 1  # Fallback to player 1


def minimax_tank(obs_state: Dict, 
                player_id: int, 
                depth: int, 
                is_maximizing: bool,
                **weights) -> Tuple[float, Optional[int]]:
    """Minimax algorithm for tank game
    
    Args:
        obs_state: Simplified game state
        player_id: Current player's ID
        depth: Search depth remaining
        is_maximizing: Whether this is a maximizing node
        **weights: Evaluation weights
    """
    if depth == 0 or obs_state['game_over']:
        score = evaluate_state(obs_state, player_id, **weights)
        return score, None
    
    best_action = None
    available_actions = get_available_tank_actions(obs_state, player_id)
    
    if is_maximizing:
        max_score = NEGA_INFI
        for action in available_actions:
            simulated_state = simulate_tank_action(obs_state, action, player_id)
            score, _ = minimax_tank(simulated_state, player_id, depth-1, False, **weights)
            if score > max_score:
                max_score, best_action = score, action
        return max_score, best_action
    else:
        min_score = POSI_INFI
        for action in available_actions:
            simulated_state = simulate_tank_action(obs_state, action, player_id)
            score, _ = minimax_tank(simulated_state, player_id, depth-1, True, **weights)
            if score < min_score:
                min_score, best_action = score, action
        return min_score, best_action

def alpha_beta_tank(obs_state: Dict,
                   player_id: int,
                   depth: int,
                   alpha: float,
                   beta: float,
                   is_maximizing: bool) -> Tuple[float, Optional[int]]:
    if depth == 0 or obs_state['game_over']:
        return evaluate_tank_state(obs_state, player_id, **DEFAULT_WEIGHTS), None
    
    best_action = None
    available_actions = get_available_tank_actions(obs_state, player_id)
    
    if is_maximizing:
        value = NEGA_INFI
        for action in available_actions:
            simulated_state = simulate_tank_action(obs_state, action, player_id)
            score, _ = alpha_beta_tank(simulated_state, player_id, depth-1, alpha, beta, False)
            if score > value:
                value, best_action = score, action
            alpha = max(alpha, value)
            #print("action,score",action,score)
            if alpha >= beta:
                break
        return value, best_action
    else:
        value = POSI_INFI
        for action in available_actions:
            simulated_state = simulate_tank_action(obs_state, action, player_id)
            score, _ = alpha_beta_tank(simulated_state, player_id, depth-1, alpha, beta, True)
            if score < value:
                value, best_action = score, action
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value, best_action

# Helper functions using only obs_state
def get_available_tank_actions(obs_state: Dict, player_id: int) -> List[int]:
    """Get list of valid actions (0-4)
    0: No action, 1: Forward, 2: Rotate left, 3: Rotate right, 4: Fire
    """
    actions = []  # Start with empty list
    
    # Find our tank
    our_tank = None
    for tank in obs_state['tanks']:
        if tank['player_id'] == player_id:
            our_tank = tank
            break
    
    if not our_tank:
        return [0]  # Only no-action if tank not found
    
    # No action is always available
    #actions.append(0)
    
    # Get tank's current position and direction
    pos = our_tank['position']
    direction = our_tank['direction']
    
    # Check forward movement
    new_pos = list(pos)
    if direction == 0:  # Up
        new_pos[1] = new_pos[1] - 1
    elif direction == 1:  # Right
        new_pos[0] =  new_pos[0] + 1
    elif direction == 2:  # Down
        new_pos[1] = new_pos[1] + 1
    elif direction == 3:  # Left
        new_pos[0] = new_pos[0] - 1
    
    # Check if new position is valid (not blocked by obstacle or other tank)
    if tuple(new_pos) not in obs_state['obstacles']:
        # Check for tank collision
        other_tank_positions = {tuple(tank['position']) for tank in obs_state['tanks'] if tank != our_tank}
        if tuple(new_pos) not in other_tank_positions and all(1 < c < len(obs_state['map']) for c in new_pos):
            #print("new_pos",new_pos)
            actions.append(1)  # Forward movement possible
    
    # Rotation is always possible
    actions.extend([2, 3])  # Left and right rotation
    
    actions.append(4)
    
    return actions

def can_move_in_direction(pos: np.ndarray, direction: float, map_data: np.ndarray) -> bool:
    """Check if tank can move in the given direction"""
    # Convert direction to grid movement
    dir_rad = direction * 2 * np.pi
    dx = np.cos(dir_rad)
    dy = np.sin(dir_rad)
    
    # Calculate new position
    new_pos = pos + np.array([dx, dy])
    
    # Convert to grid coordinates
    grid_x = int(new_pos[0] * map_data.shape[0])
    grid_y = int(new_pos[1] * map_data.shape[1])
    
    # Check bounds
    if not (0 <= grid_x < map_data.shape[0] and 0 <= grid_y < map_data.shape[1]):
        return False
    
    # Check for obstacles (steel walls are at channel 2)
    return not map_data[grid_y, grid_x, 2]

def simulate_tank_action(obs_state: Dict, action: int, player_id: int) -> Dict:
    """Simulate the effect of an action in the simplified game
    
    Args:
        obs_state: Current game state
        action: Action to simulate (0-4)
        player_id: ID of the acting player
    
    Returns:
        New game state after action
    """
    # Deep copy the state
    new_state = {
        'map': [row[:] for row in obs_state['map']],
        'tanks': [tank.copy() for tank in obs_state['tanks']],
        'bullets': [bullet[:] for bullet in obs_state['bullets']],
        'obstacles': set(obs_state['obstacles']),
        'game_over': obs_state['game_over'],
        'winner': obs_state['winner']
    }
    
    # Find our tank
    our_tank = None
    for tank in new_state['tanks']:
        if tank['player_id'] == player_id:
            our_tank = tank
            break
    
    if not our_tank:
        return new_state
    
    # Update tank state based on action
    if action == 1:  # Forward
        new_pos = list(our_tank['position'])
        direction = our_tank['direction']
        
        if direction == 0:  # Up
            new_pos[1] = max(0, new_pos[1] - 1)
        elif direction == 1:  # Right
            new_pos[0] = min(len(new_state['map'][0]) - 1, new_pos[0] + 1)
        elif direction == 2:  # Down
            new_pos[1] = min(len(new_state['map']) - 1, new_pos[1] + 1)
        elif direction == 3:  # Left
            new_pos[0] = max(0, new_pos[0] - 1)
        
        # Check if new position is valid
        if tuple(new_pos) not in new_state['obstacles']:
            # Check for tank collision
            other_tank_positions = {tuple(tank['position']) for tank in new_state['tanks'] if tank != our_tank}
            if tuple(new_pos) not in other_tank_positions:
                our_tank['position'] = new_pos
            
    elif action == 2:  # Rotate left
        our_tank['direction'] = (our_tank['direction'] - 1) % 4
        
    elif action == 3:  # Rotate right
        our_tank['direction'] = (our_tank['direction'] + 1) % 4
        
    elif action == 4:  # Fire
        # Add new bullet if we don't exceed bullet limit
        our_bullets = sum(1 for bullet in new_state['bullets'] if bullet[3] == player_id)
        if our_bullets < 4:  # Max 4 bullets per tank
            new_bullet = [
                our_tank['position'][0],
                our_tank['position'][1],
                our_tank['direction'],
                player_id
            ]
            new_state['bullets'].append(new_bullet)
    
    # Simulate bullet movement and collisions
    updated_bullets = []
    for bullet in new_state['bullets']:
        new_bullet_pos = [bullet[0], bullet[1]]
        bullet_dir = bullet[2]
        
        # Move bullet
        if bullet_dir == 0:  # Up
            new_bullet_pos[1] -= 1
        elif bullet_dir == 1:  # Right
            new_bullet_pos[0] += 1
        elif bullet_dir == 2:  # Down
            new_bullet_pos[1] += 1
        elif bullet_dir == 3:  # Left
            new_bullet_pos[0] -= 1
        
        # Check if bullet hits anything
        bullet_tuple_pos = tuple(new_bullet_pos)
        
        # Check map boundaries
        if not (0 <= new_bullet_pos[0] < len(new_state['map'][0]) and 
                0 <= new_bullet_pos[1] < len(new_state['map'])):
            continue
        
        # Check obstacle hits
        if bullet_tuple_pos in new_state['obstacles']:
            new_state['obstacles'].remove(bullet_tuple_pos)
            continue
        
        # Check tank hits
        hit_tank = False
        for tank in new_state['tanks']:
            if tuple(tank['position']) == bullet_tuple_pos:
                if bullet[3] != tank['player_id']:  # Only count hits on enemy tank
                    new_state['game_over'] = True
                    new_state['winner'] = bullet[3]  # Bullet owner wins
                hit_tank = True
                break
        
        if not hit_tank:
            # Update bullet position and keep it
            bullet[0], bullet[1] = new_bullet_pos
            updated_bullets.append(bullet)
    
    new_state['bullets'] = updated_bullets
    
    return new_state

def is_tank_action_valid(obs_state: Dict, action: int, player_id: int) -> bool:
    """Check if an action is valid in the current state"""
    if action == 0:  # No action is always valid
        return False
    
    mapsize = obs_state.get('mapsize', 6)
    if action == 1:  # Forward
        tank = obs_state['tanks'][player_id]
        grid_pos = np.array(tank['position']) * (mapsize-1)
        direction = int(tank['direction'] * 4) % 4
        direction_offsets = np.array([
            [0, -1],  # direction 0 (up)
            [1, 0],   # direction 1 (right)
            [0, 1],   # direction 2 (down)
            [-1, 0]   # direction 3 (left)
        ])
        target = grid_pos + direction_offsets[direction]
        target = tuple(target.astype(int))
        return (target not in obs_state.get('obstacles', []) and 
                all(0 <= c < mapsize for c in target))
    elif action == 4:  # Fire - check bullet limit
        return len(obs_state.get('bullets', [])) < 4  # Assuming max 4 bullets allowed
    return True  # Other actions (2,3) are always valid

def evaluate_tank_state(obs_state: Dict, player_id: int, **weights) -> float:
    """Evaluation function using the evaluation module"""
    score = evaluate_state(obs_state, player_id, **weights)
    #print("score is",score)
    return score

def is_game_over(obs_state: Dict) -> bool:
    """Check if the game is over based on tank health"""
    tanks_data = obs_state['tanks']
    # Check if either tank has 0 health (health is at index 4)
    return tanks_data[4] <= 0 or tanks_data[9] <= 0  # 4 for player 1, 9 for player 2
