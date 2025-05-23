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
        tanks = []
        for t in raw_tanks:
            if 'position' not in t:
                # 简化格式
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
            # 打印位置信息和朝向
            px, py = my_tank['position']
            ex, ey = enemy_tank['position']
            pd = my_tank['direction'] * 90
            ed = enemy_tank['direction'] * 90
            print(f"- 自身位置: ({px:.2f}, {py:.2f})")
            print(f"- 自身朝向: {pd:.1f}度")
            print(f"- 敌人位置: ({ex:.2f}, {ey:.2f})")
            print(f"- 敌人朝向: {ed:.1f}度")
        
        # 创建求解器
        self.solver = Solver()
        self.vars = {}
        self._init_vars()
        
        # 构建约束条件
        clauses = []
        
        # 躲避子弹约束
        # self._build_dodge_clauses(state)
        
        # 追踪敌人约束
        next_action= self._build_chase_clauses(state)
        if next_action==[[self._action_to_var("forward")]] and self._is_blocked(state):
            clauses.extend([self._action_to_var("fire")])
        else:
            clauses.extend(next_action)
        
        # 添加所有约束到求解器
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
        
        # 求解
        if self.solver.check() == sat:
            model = self.solver.model()
            action = self._extract_action(model)
        else:
            # 在求解失败时，打印求解器的状态信息和无法满足的约束
            print("- 求解失败，检查以下约束：")
            for clause in self.solver.assertions():
                print(f"  约束: {clause}")
            print("求解器状态：")
            print(self.solver.check())  # 打印求解失败的原因
            action = np.random.randint(0, 5)  # 在无解情况下随机选择动作
            print(f"- 随机选择动作: {self._action_to_str(action)}")
        
        if self.debug:
            print(f"3. 选择动作: {self._action_to_str(action)}")
        
        return action
    
    def _init_vars(self):
        """初始化变量"""
        # 为每个动作创建布尔变量
        actions = ["stay", "forward", "turn_left", "turn_right", "fire"]
        self.action_vars = []
        for action in actions:
            var = Bool(action)
            self.vars[action] = var
            self.action_vars.append(var)
    
    # def _build_dodge_clauses(state: Dict) -> List:
    #     clauses = []
    #     #判断是否有敌方子弹
    #     bullets=state['bullets']
    #     for bullet in bullet:
    #         #写死敌方id
    #         if bullet[2]==1:
                #如果在直线上并且只有两格了
                
    def _build_chase_clauses(self, state: Dict) -> List:
        """构建追踪敌人的约束"""
        clauses = []
        tanks = state['tanks']
        my_tank = tanks[1]  # player_id为2的坦克
        enemy_tank = tanks[0]  # player_id为1的坦克
        
        # 计算与敌人的坐标差值
        dx = enemy_tank['position'][0] - my_tank['position'][0]  # 敌人x - 自己x
        dy = enemy_tank['position'][1] - my_tank['position'][1]  # 敌人y - 自己y
        
        # 检查自己当前朝向（0-上，1-右，2-下，3-左）
        tank_angle = my_tank['direction']
        # 判断哪个坐标差值更小，选择那个轴进行移动
        if abs(dx) < abs(dy):
            # 走向同一竖线
            if dx > 0:
                #通过选转，最后朝右
                if tank_angle==0:
                    clauses.append([self._action_to_var("turn_right")])  # 向右转
                elif tank_angle==1:
                    clauses.append([self._action_to_var("forward")])
                else:
                    clauses.append([self._action_to_var("turn_left")])  # 向左转
            elif dx < 0:
                #通过选转，最后朝左
                if tank_angle==0:
                    clauses.append([self._action_to_var("turn_left")])  # 向左转
                elif tank_angle==3:
                    clauses.append([self._action_to_var("forward")])
                else:
                    clauses.append([self._action_to_var("turn_right")])  
            else:
                 #检查是不是在射击方向上
                if dy > 0:
                    #需要朝下
                    if tank_angle==3:
                        clauses.append([self._action_to_var("turn_left")])  # 向左转
                    elif tank_angle==2:
                        clauses.append([self._action_to_var("fire")])
                    else:
                        clauses.append([self._action_to_var("turn_right")])  
                else :
                    #需要朝上
                    if tank_angle==1:
                        clauses.append([self._action_to_var("turn_left")])  # 向左转
                    elif tank_angle==0:
                        clauses.append([self._action_to_var("fire")])
                    else:
                        clauses.append([self._action_to_var("turn_right")])  
        else:
            # 走向统一横线
            if dy > 0:
                # 需要朝下
                if tank_angle == 3:  # 朝上
                    clauses.append([self._action_to_var("turn_left")])  # 向左转
                elif tank_angle == 2:  # 朝下
                    clauses.append([self._action_to_var("forward")])  # 向前
                else:
                    clauses.append([self._action_to_var("turn_right")])  # 向右转
            elif dy < 0:
                # 需要朝上
                if tank_angle == 1:  # 朝右
                    clauses.append([self._action_to_var("turn_left")])  # 向左转
                elif tank_angle == 0:  # 朝上
                    clauses.append([self._action_to_var("forward")])  # 向前
                else:
                    clauses.append([self._action_to_var("turn_right")])  # 向右转
            else:
                 #检查是不是在射击方向上
                if dx > 0:
                    #需要朝右
                    if tank_angle==2:
                        clauses.append([self._action_to_var("turn_left")])  # 向左转
                    elif tank_angle==1:
                        clauses.append([self._action_to_var("fire")])
                    else:
                        clauses.append([self._action_to_var("turn_right")])  
                else :
                    #需要朝左
                    if tank_angle==0:
                        clauses.append([self._action_to_var("turn_left")])  # 向左转
                    elif tank_angle==3:
                        clauses.append([self._action_to_var("fire")])
                    else:
                        clauses.append([self._action_to_var("turn_right")]) 
        
        
        return clauses
   

    def _is_blocked(self, state: Dict) -> bool:
        """判断前方是否有障碍物"""
        tanks = state['tanks']
        my_tank = tanks[1]  # player_id为2的坦克
        current_x, current_y = my_tank['position'][0], my_tank['position'][1]
        current_angle = my_tank['direction']  # 获取当前坦克位置和朝向（0-上，1-右，2-下，3-左）
        
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
