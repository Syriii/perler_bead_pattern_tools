"""
颜色工具类
提供颜色转换、距离计算和调色板匹配功能
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from .palettes import ALL_PALETTES


class ColorUtils:
    """颜色处理工具类"""
    
    def __init__(self):
        self._color_cache = {}  # 颜色匹配缓存
        
    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """将十六进制颜色转换为RGB"""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            raise ValueError(f"Invalid hex color: {hex_color}")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    
    
    @staticmethod
    def color_distance(color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
        """
        计算两个RGB颜色之间的距离
        使用加权欧几里得距离，考虑人眼对不同颜色的敏感度
        """
        r1, g1, b1 = color1
        r2, g2, b2 = color2
        
        # 加权系数，人眼对绿色最敏感，红色次之，蓝色最不敏感
        weight_r = 0.3
        weight_g = 0.59
        weight_b = 0.11
        
        distance = np.sqrt(
            weight_r * (r1 - r2) ** 2 +
            weight_g * (g1 - g2) ** 2 +
            weight_b * (b1 - b2) ** 2
        )
        
        return distance
    
    def find_closest_palette_color(self, target_rgb: Tuple[int, int, int], 
                                 palette_name: str) -> Dict[str, any]:
        """
        在指定调色板中找到最接近的颜色
        
        Args:
            target_rgb: 目标RGB颜色
            palette_name: 调色板名称
            
        Returns:
            包含颜色信息的字典: {'name': str, 'color': str, 'rgb': tuple, 'distance': float}
        """
        # 检查缓存
        cache_key = (target_rgb, palette_name)
        if cache_key in self._color_cache:
            return self._color_cache[cache_key]
        
        palette = ALL_PALETTES.get(palette_name, [])
        if not palette:
            raise ValueError(f"Unknown palette: {palette_name}")
        
        min_distance = float('inf')
        closest_color = None
        
        for color_info in palette:
            palette_rgb = self.hex_to_rgb(color_info['color'])
            distance = self.color_distance(target_rgb, palette_rgb)
            
            if distance < min_distance:
                min_distance = distance
                closest_color = {
                    'name': color_info['name'],
                    'color': color_info['color'],
                    'rgb': palette_rgb,
                    'distance': distance
                }
        
        # 缓存结果
        self._color_cache[cache_key] = closest_color
        return closest_color
    
    def generate_palette_map(self, palette_name: str) -> Dict[str, Tuple[int, int, int]]:
        """
        生成调色板映射表
        
        Args:
            palette_name: 调色板名称
            
        Returns:
            颜色名称到RGB的映射字典
        """
        palette = ALL_PALETTES.get(palette_name, [])
        if not palette:
            raise ValueError(f"Unknown palette: {palette_name}")
        
        palette_map = {}
        for color_info in palette:
            palette_map[color_info['name']] = self.hex_to_rgb(color_info['color'])
        
        return palette_map
    
    
    
    def clear_cache(self):
        """清空颜色匹配缓存"""
        self._color_cache.clear()