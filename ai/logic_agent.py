import numpy as np
from typing import Dict, List, Tuple
from z3 import Solver, Or, Bool, sat, Not

class LogicAgent:
    """基于逻辑规划的AI智能体"""
    
    def __init__(self):
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
            print(f"- 自身位置: ({tank_data[0]:.2f}, {tank_data[1]:.2f})")
            print(f"- 自身朝向: {tank_data[2] * 360:.1f}度")
            print(f"- 炮塔朝向: {tank_data[3] * 360:.1f}度")
            print(f"- 敌人位置: ({tank_data[7]:.2f}, {tank_data[8]:.2f})")
        
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
        
        # 攻击约束
        clauses.extend(self._build_attack_clauses(state))
        
        # 躲避约束
        clauses.extend(self._build_dodge_clauses(state))
        
        # 添加所有约束到求解器
        if self.debug:
            print("2. 生成约束条件:")
            print(f"- 总约束数: {len(clauses)}")
        
        for clause in clauses:
            self.solver.add(Or(clause))
        
        if self.debug:
            print("求解器状态:")
            print(f"- 约束数量: {len(self.solver.assertions())}")
        
        # 求解
        if self.solver.check() == sat:
            model = self.solver.model()
            action = self._extract_action(model)
        else:
            if self.debug:
                print("- 求解失败，使用随机动作")
            # 在无解情况下随机选择动作
            action = np.random.randint(0, 8)
            print(f"- 随机选择动作: {self._action_to_str(action)}")
        
        if self.debug:
            print(f"3. 选择动作: {self._action_to_str(action)}")
        
        return action
    
    def _init_vars(self):
        """初始化变量"""
        # 为每个动作创建布尔变量
        actions = ["stay", "forward", "backward", "turn_left", "turn_right", 
                  "turret_left", "turret_right", "fire"]
        self.action_vars = []
        for action in actions:
            var = Bool(action)
            self.vars[action] = var
            self.action_vars.append(var)
    
    def _build_basic_clauses(self) -> List:
        """构建基本约束：每次只能执行一个动作"""
        clauses = []
        # 至少执行一个动作
        clauses.append(self.action_vars)
        # 不能同时执行多个动作
        for i, var1 in enumerate(self.action_vars):
            for j, var2 in enumerate(self.action_vars):
                if i < j:
                    clauses.append([Not(var1), Not(var2)])
        return clauses
    
    def _build_chase_clauses(self, state: Dict) -> List:
        """构建追踪敌人的约束"""
        clauses = []
        tank_data = state['tanks']
        
        # 计算与敌人的距离和角度
        dx = tank_data[7] - tank_data[0]  # 敌人x - 自己x
        dy = tank_data[8] - tank_data[1]  # 敌人y - 自己y
        distance = np.sqrt(dx*dx + dy*dy)
        target_angle = np.arctan2(dy, dx) * 180 / np.pi
        if target_angle < 0:
            target_angle += 360
            
        current_angle = tank_data[2] * 360  # 当前朝向
        turret_angle = tank_data[3] * 360   # 炮塔朝向
        
        # 计算需要转向的角度
        angle_diff = (target_angle - current_angle) % 360
        turret_diff = (target_angle - turret_angle) % 360
        
        # 根据距离决定行为
        if distance > 0.4:  # 如果距离较远，追击
            if abs(angle_diff) > 20 and abs(angle_diff - 360) > 20:
                # 需要转向
                if angle_diff < 180:
                    clauses.append([self._action_to_var("turn_right")])
                else:
                    clauses.append([self._action_to_var("turn_left")])
            else:
                # 角度合适，前进
                clauses.append([self._action_to_var("forward")])
        elif distance < 0.2:  # 如果距离太近，后退
            clauses.append([self._action_to_var("backward")])
        
        # 调整炮塔朝向
        if abs(turret_diff) > 10 and abs(turret_diff - 360) > 10:
            if turret_diff < 180:
                clauses.append([self._action_to_var("turret_right")])
            else:
                clauses.append([self._action_to_var("turret_left")])
        
        return clauses
    
    def _build_attack_clauses(self, state: Dict) -> List:
        """构建攻击约束"""
        clauses = []
        tank_data = state['tanks']
        
        # 计算炮塔与敌人的角度差
        dx = tank_data[7] - tank_data[0]
        dy = tank_data[8] - tank_data[1]
        target_angle = np.arctan2(dy, dx) * 180 / np.pi
        if target_angle < 0:
            target_angle += 360
            
        turret_angle = tank_data[3] * 360
        angle_diff = (target_angle - turret_angle) % 360
        
        # 如果炮塔朝向接近目标，开火
        if abs(angle_diff) < 10 or abs(angle_diff - 360) < 10:
            clauses.append([self._action_to_var("fire")])
        
        return clauses
    
    def _build_dodge_clauses(self, state: Dict) -> List:
        """构建躲避约束"""
        clauses = []
        # TODO: 实现躲避子弹的逻辑
        return clauses
    
    def _action_to_var(self, action: str) -> bool:
        """动作名称转变量"""
        return self.vars[action]
    
    def _extract_action(self, model) -> int:
        """从模型中提取选择的动作"""
        action_map = {
            "stay": 0,
            "forward": 1,
            "backward": 2,
            "turn_left": 3,
            "turn_right": 4,
            "turret_left": 5,
            "turret_right": 6,
            "fire": 7
        }
        
        for action, var in self.vars.items():
            if model[var]:
                return action_map[action]
        return 0  # 默认不动
    
    def _action_to_str(self, action: int) -> str:
        """动作ID转字符串"""
        action_strs = ["stay", "forward", "backward", "turn_left", "turn_right",
                      "turret_left", "turret_right", "fire"]
        return action_strs[action]
