from abc import ABC, abstractmethod
from typing import Dict

class BaseAgent(ABC):
    """智能体基类，定义了所有智能体必须实现的接口"""
    
    @abstractmethod
    def select_action(self, state: Dict, training: bool = True) -> int:
        """选择动作
        
        Args:
            state: 游戏状态，包含地图、坦克和子弹信息
            training: 是否处于训练模式
            
        Returns:
            action: 选择的动作（0-5）
            0: 不动
            1: 前进
            2: 左转
            3: 右转
            4: 开火
            5: 后退
        """
        pass
    
    def train(self):
        """训练智能体
        
        默认实现为空，只有需要训练的智能体（如DQN）才需要重写此方法
        """
        pass
    
    def save(self, path: str):
        """保存模型
        
        默认实现为空，只有需要保存模型的智能体才需要重写此方法
        
        Args:
            path: 模型保存路径
        """
        pass
    
    def load(self, path: str):
        """加载模型
        
        默认实现为空，只有需要加载模型的智能体才需要重写此方法
        
        Args:
            path: 模型加载路径
        """
        pass