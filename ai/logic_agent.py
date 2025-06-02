import numpy as np
from typing import Dict, List
from z3 import Solver, Or, Bool, sat

from ai.base_agent import BaseAgent

class LogicAgent(BaseAgent):
    """基于逻辑规划的AI智能体"""
    
    def __init__(self):
        self.debug = True
        self.solver = None
        self.vars = {}
        self.action_vars = []
        
    def select_action(self, state: Dict) -> int:
        """选择动作
        
        Args:
            state: 游戏状态字典，包含 'map'、'tanks' 和 'bullets' 等信息
        Returns:
            action: 选择的动作（0-4，分别表示 stay、forward、turn_left、turn_right、fire）
        """
        # 兼容简化环境的state格式，将简化版tank属性(x,y,angle)映射为(position,direction)
        raw_tanks = state.get('tanks', [])
        # print("raw_tanks:", raw_tanks)
        tanks = []
        for t in raw_tanks:
            if 'position' not in t:
                tanks.append({
                    'position': [t.get('x', 0), t.get('y', 0)],
                    'direction': t.get('angle', t.get('direction', 0)),
                    'player_id': t.get('player_id', None)
                })
            else:
                tanks.append(t)
        state['tanks'] = tanks
        if self.debug:
            print("\nLogicAgent决策过程:")
            print("1. 当前状态(tanks列表):", tanks)
            # 假设第二个对象是AI控制的坦克（player_id=2）
            my_tank = tanks[1]
            enemy_tank = tanks[0]
            px, py = my_tank['position']
            ex, ey = enemy_tank['position']
            pd = my_tank['direction'] * 90
            ed = enemy_tank['direction'] * 90
            print(f"- 自身位置: ({px:.2f}, {py:.2f})")
            print(f"- 自身朝向: {pd:.1f}度")
            print(f"- 敌人位置: ({ex:.2f}, {ey:.2f})")
            print(f"- 敌人朝向: {ed:.1f}度")
        
        self.solver = Solver()
        self.vars = {}
        self._init_vars()
        
        clauses = []
        # 追踪敌人约束
        next_action= self._build_chase_clauses(state)
        if next_action==[[self._action_to_var("forward")]] and self._is_blocked(state):
            clauses.extend([self._action_to_var("fire")])
        else:
            clauses.extend(next_action)
        
        if self.debug:
            print("2. 生成约束条件:")
            print(f"- 总约束数: {len(clauses)}")
        
        for clause in clauses:
            self.solver.add(Or(clause))
            if self.debug:
                print(f"  添加约束: {clause}")
        
        if self.debug:
            print("求解器状态:")
            print(f"- 约束数量: {len(self.solver.assertions())}")
        
        if self.solver.check() == sat:
            model = self.solver.model()
            action = self._extract_action(model)
        else:
            print("- 求解失败，检查以下约束：")
            for clause in self.solver.assertions():
                print(f"  约束: {clause}")
            print("求解器状态：")
            print(self.solver.check())  
            action = np.random.randint(0, 5) 
            print(f"- 随机选择动作: {self._action_to_str(action)}")
        
        if self.debug:
            print(f"3. 选择动作: {self._action_to_str(action)}")
        
        return action
    
    def _init_vars(self):
        """初始化变量"""
        actions = ["stay", "forward", "turn_left", "turn_right", "fire"]
        self.action_vars = []
        for action in actions:
            var = Bool(action)
            self.vars[action] = var
            self.action_vars.append(var)
    
                
    def _build_chase_clauses(self, state: Dict) -> List:
        """构建追踪敌人的约束"""
        clauses = []
        tanks = state['tanks']
        my_tank = tanks[1]  # player_id为2的坦克
        enemy_tank = tanks[0]  # player_id为1的坦克
        
        # 计算与敌人的坐标差值
        dx = enemy_tank['position'][0] - my_tank['position'][0] 
        dy = enemy_tank['position'][1] - my_tank['position'][1] 
        
        # 检查自己当前朝向（0-上，1-右，2-下，3-左）
        tank_angle = my_tank['direction']
        # 判断哪个坐标差值更小，选择那个轴进行移动
        if abs(dx) < abs(dy):
            # 走向同一竖线
            if dx > 0:
                #通过旋转，最后朝右
                if tank_angle==0:
                    clauses.append([self._action_to_var("turn_right")])  
                elif tank_angle==1:
                    clauses.append([self._action_to_var("forward")])
                else:
                    clauses.append([self._action_to_var("turn_left")])  
            elif dx < 0:
                #通过旋转，最后朝左
                if tank_angle==0:
                    clauses.append([self._action_to_var("turn_left")]) 
                elif tank_angle==3:
                    clauses.append([self._action_to_var("forward")])
                else:
                    clauses.append([self._action_to_var("turn_right")])  
            else:
                 #检查是不是在射击方向上
                if dy > 0:
                    #需要朝下
                    if tank_angle==3:
                        clauses.append([self._action_to_var("turn_left")])  
                    elif tank_angle==2:
                        clauses.append([self._action_to_var("fire")])
                    else:
                        clauses.append([self._action_to_var("turn_right")])  
                else :
                    #需要朝上
                    if tank_angle==1:
                        clauses.append([self._action_to_var("turn_left")]) 
                    elif tank_angle==0:
                        clauses.append([self._action_to_var("fire")])
                    else:
                        clauses.append([self._action_to_var("turn_right")])  
        else:
            # 走向统一横线
            if dy > 0:
                # 需要朝下
                if tank_angle == 3:  
                    clauses.append([self._action_to_var("turn_left")])  
                elif tank_angle == 2:  
                    clauses.append([self._action_to_var("forward")]) 
                else:
                    clauses.append([self._action_to_var("turn_right")])  
            elif dy < 0:
                # 需要朝上
                if tank_angle == 1: 
                    clauses.append([self._action_to_var("turn_left")])
                elif tank_angle == 0:  
                    clauses.append([self._action_to_var("forward")])  
                else:
                    clauses.append([self._action_to_var("turn_right")]) 
            else:
                 #检查是不是在射击方向上
                if dx > 0:
                    #需要朝右
                    if tank_angle==2:
                        clauses.append([self._action_to_var("turn_left")])  
                    elif tank_angle==1:
                        clauses.append([self._action_to_var("fire")])
                    else:
                        clauses.append([self._action_to_var("turn_right")])  
                else :
                    #需要朝左
                    if tank_angle==0:
                        clauses.append([self._action_to_var("turn_left")]) 
                    elif tank_angle==3:
                        clauses.append([self._action_to_var("fire")])
                    else:
                        clauses.append([self._action_to_var("turn_right")]) 
        
        
        return clauses
   

    def _is_blocked(self, state: Dict) -> bool:
        """判断前方是否有障碍物"""
        tanks = state['tanks']
        my_tank = tanks[1]  
        current_x, current_y = my_tank['position'][0], my_tank['position'][1]
        current_angle = my_tank['direction']  
        
        target_x = current_x 
        target_y = current_y 
        if current_angle==0:
            target_y=target_y-1
        elif current_angle==1:
            target_x=target_x+1
        elif current_angle==2:
            target_y=target_y+1
        elif current_angle==3:
            target_x=target_x-1
        # 获取障碍物集合
        obstacles = state.get('obstacles', [])  # 从状态中获取障碍物列表
        
        # 检查前方目标位置是否有障碍物
        target_pos = (target_x, target_y)
        if target_pos in obstacles:
            return True
        return False
    
    def _action_to_var(self, action: str) -> bool:
        """动作名称转变量"""
        return self.vars[action]
    
    def _extract_action(self, model) -> int:
        """从模型中提取选择的动作"""
        action_map = {
            "stay": 0,
            "forward": 1,
            "turn_left": 2,
            "turn_right": 3,
            "fire": 4
        }
        
        for action, var in self.vars.items():
            if model[var]:
                return action_map[action]
        return 0  # 默认不动
    
    def _action_to_str(self, action: int) -> str:
        """动作ID转字符串"""
        action_strs = ["stay", "forward",  "turn_left", "turn_right", "fire"]
        return action_strs[action]

class LogicExpertAgent(LogicAgent):
    """专家逻辑智能体，从玩家0（RL智能体）的视角给出建议"""
    
    def __init__(self):
        super().__init__()
        self.debug = False  # 默认关闭调试，避免专家输出过多信息
    
    def select_action(self, state: Dict) -> int:
        """从RL智能体视角选择动作
        
        Args:
            state: 游戏状态字典
        Returns:
            action: 专家建议的动作（0-4，不包含后退）
        """
        # 兼容简化环境的state格式，将简化版tank属性(x,y,angle)映射为(position,direction)
        raw_tanks = state.get('tanks', [])
        tanks = []
        for t in raw_tanks:
            if 'position' not in t:
                tanks.append({
                    'position': [t.get('x', 0), t.get('y', 0)],
                    'direction': t.get('angle', t.get('direction', 0)),
                    'player_id': t.get('player_id', None)
                })
            else:
                tanks.append(t)
        
        if len(tanks) < 2:
            return 0  # 如果状态异常，返回停留动作
        
        # 创建交换后的状态
        swapped_state = state.copy()
        
        # 深拷贝坦克列表，以免修改原始状态
        swapped_tanks = []
        # 将坦克0和坦克1交换位置，使原本的坦克1（对手）变成专家视角下的自己（索引1）
        for i, tank in enumerate(tanks):
            new_tank = tank.copy()
            if i == 0:
                new_tank['player_id'] = 1  # 将原来的坦克0（RL智能体）变为坦克1（对手）
                swapped_tanks.append(new_tank)
            elif i == 1:
                new_tank['player_id'] = 0  # 将原来的坦克1（对手）变为坦克0（自己）
                swapped_tanks.insert(0, new_tank)  # 插入到队首
            else:
                swapped_tanks.append(new_tank)
        
        swapped_state['tanks'] = swapped_tanks
        
        # 如果有bullets字段，也需要交换子弹的所有者标识
        if 'bullets' in state:
            swapped_bullets = []
            for bullet in state.get('bullets', []):
                new_bullet = bullet.copy()
                if 'owner' in new_bullet:
                    if new_bullet['owner'] == 0:
                        new_bullet['owner'] = 1
                    elif new_bullet['owner'] == 1:
                        new_bullet['owner'] = 0
                swapped_bullets.append(new_bullet)
            swapped_state['bullets'] = swapped_bullets
            
        # 使用父类方法计算动作，但使用交换后的状态
        logic_action = super().select_action(swapped_state)
        
        # LogicAgent的动作空间: 0-stay, 1-forward, 2-turn_left, 3-turn_right, 4-fire
        # 这些对于专家建议来说是合适的动作，直接返回
        return logic_action
        
        # 如果状态异常，返回停留动作
        return 0
