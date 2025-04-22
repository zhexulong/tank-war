import os
import json
import time
import datetime
from typing import Dict, List, Any

from game.config import DATA_COLLECTION

class DataCollector:
    """数据收集器，负责收集和保存游戏数据"""
    
    def __init__(self):
        # 确保数据保存目录存在
        self.save_directory = DATA_COLLECTION["save_directory"]
        os.makedirs(self.save_directory, exist_ok=True)
        
        # 初始化数据列表
        self.collected_data = []
        
        # 会话ID（使用时间戳）
        self.session_id = int(time.time())
    
    def collect_data(self, game_data: Dict[str, Any]):
        """收集一局游戏的数据"""
        # 添加时间戳
        game_data["timestamp"] = datetime.datetime.now().isoformat()
        game_data["session_id"] = self.session_id
        
        # 添加到数据列表
        self.collected_data.append(game_data)
        
        # 保存数据
        self.save_data()
    
    def save_data(self):
        """保存收集的数据"""
        # 生成文件名
        filename = f"game_data_{self.session_id}.json"
        file_path = os.path.join(self.save_directory, filename)
        
        # 保存为JSON文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.collected_data, f, ensure_ascii=False, indent=2)
    
    def get_collected_data(self) -> List[Dict[str, Any]]:
        """获取收集的数据"""
        return self.collected_data
    
    def clear_data(self):
        """清空收集的数据"""
        self.collected_data = []
    
    def analyze_data(self) -> Dict[str, Any]:
        """分析收集的数据，计算统计信息"""
        if not self.collected_data:
            return {}
        
        # 初始化统计信息
        stats = {
            "total_games": len(self.collected_data),
            "avg_game_time": 0,
            "player1_wins": 0,
            "player2_wins": 0,
            "draws": 0,
            "avg_kills_per_game": 0,
            "avg_hits_per_game": 0
        }
        
        # 计算统计信息
        total_time = 0
        total_kills = 0
        total_hits = 0
        
        for game in self.collected_data:
            total_time += game.get("time", 0)
            
            if game.get("winner") == 1:
                stats["player1_wins"] += 1
            elif game.get("winner") == 2:
                stats["player2_wins"] += 1
            else:
                stats["draws"] += 1
            
            total_kills += game.get("player1_kills", 0) + game.get("player2_kills", 0)
            total_hits += game.get("player1_hits", 0) + game.get("player2_hits", 0)
        
        # 计算平均值
        stats["avg_game_time"] = total_time / stats["total_games"]
        stats["avg_kills_per_game"] = total_kills / stats["total_games"]
        stats["avg_hits_per_game"] = total_hits / stats["total_games"]
        
        return stats