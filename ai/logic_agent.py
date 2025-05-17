import numpy as np
from typing import Dict, List
from z3 import Solver, Or, Bool, sat, Not
import math

class LogicAgent:
    """基于逻辑规划的AI智能体"""
    
    def __init__(self, game_manager):
        self.game_manager = game_manager  # 确保传入 game_manager
        self.debug = True
        self.solver = None
        self.vars = {}
        self.action_vars = []
        
    def select_action(self, state: Dict, epsilon: float = 0.0) -> int:
        """选择动作
        
        Args:
            state: 游戏状态
            epsilon: 随机探索概率（未使用）        
        Returns:
            action: 选择的动作（0-7）
        """
        if self.debug:
            print("\nLogicAgent决策过程:")
            print("1. 当前状态:")
            tank_data = state['tanks']
            print(tank_data)
            print(f"- 自身位置: ({tank_data[0]:.2f}, {tank_data[1]:.2f})")
            print(f"- 自身朝向: {tank_data[2] * 360:.1f}度")
            print(f"- 敌人位置: ({tank_data[3]:.2f}, {tank_data[4]:.2f})")
            print(f"- 敌人朝向: {tank_data[5] * 360:.1f}度")
        
        # 创建求解器
        self.solver = Solver()
        self.vars = {}
        self._init_vars()
        
        # 构建约束条件
        clauses = []
        
        # 基本约束：每次只能执行一个动作
        clauses.extend(self._build_basic_clauses())
        
        # 追踪敌人约束
        clauses.extend(self._build_chase_clauses(state))
        
        # # 攻击约束
        # clauses.extend(self._build_attack_clauses(state))
        
        # # 道路清除约束
        # clauses.extend(self._build_clear_path_clauses(state))
        
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
    
    def _build_basic_clauses(self) -> List:
        """构建基本约束：每次只能执行一个动作"""
        clauses = []
        
        # 确保只能选择一个动作
        clauses.append(Or(self.action_vars))  # 选择一个动作
        
        return clauses
    
    def _build_chase_clauses(self, state: Dict) -> List:
        """构建追踪敌人的约束"""
        clauses = []
        tank_data = state['tanks']
        
        # 计算与敌人的坐标差值
        dx = tank_data[3] - tank_data[0]  # 敌人x - 自己x
        dy = tank_data[4] - tank_data[1]  # 敌人 y - 自己 y
        
        #检查自己当前朝向
        tank_angle = tank_data[2]*4
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
                if tank_data[2] == 1:  # 朝右
                    clauses.append([self._action_to_var("turn_left")])  # 向左转
                elif tank_data[2] == 0:  # 朝上
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
    
    def _build_attack_clauses(self, state: Dict) -> List:
        """构建攻击约束"""
        clauses = []
        tank_data = state['tanks']
        
        # 计算炮塔与敌人的角度差
        dx = tank_data[3] - tank_data[0]
        dy = tank_data[4] - tank_data[1]
        target_angle = np.arctan2(dy, dx) * 180 / np.pi
        if target_angle < 0:
            target_angle += 360
            
        angle_diff = (target_angle - tank_data[5]) % 360
        
        # 如果炮塔朝向接近目标，开火
        if abs(angle_diff) < 10:  # 向敌人射击
            clauses.append([self._action_to_var("fire")])
        
        return clauses

    def _build_clear_path_clauses(self, state: Dict) -> List:
        """构建清除障碍物的约束"""
        clauses = []
        tank_data = state['tanks']
        
        # 假设障碍物是可以被射击的，检查前方是否有障碍物
        if self._is_blocked(state):  # 假设存在前方阻挡
            clauses.append([self._action_to_var("fire")])  # 射击前方
        
        return clauses

    def _is_blocked(self, state: Dict) -> bool:
        """判断前方是否有障碍物"""
        import math
        tank_data = state['tanks']
        current_x, current_y, current_angle = tank_data[0], tank_data[1], tank_data[2] * 360  # 获取当前坦克位置和朝向
        
        # 计算坦克前方一小步的坐标
        step_length = 1  # 步长
        angle_rad = math.radians(current_angle)  # 转为弧度
        
        dx = step_length * math.cos(angle_rad)
        dy = step_length * math.sin(angle_rad)
        
        target_x = current_x + dx
        target_y = current_y + dy
        
        # 获取障碍物集合
        obstacles = self.game_manager.current_map.obstacles  # 获取障碍物的集合
        
        target_x_int = int(target_x)  # 确保是整数索引
        target_y_int = int(target_y)  # 确保是整数索引
        
        # 检查前方目标位置是否有障碍物
        if (target_x_int, target_y_int) in obstacles:
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
