"""
Export utilities for Perler Bead Pattern Tool
拼豆图案工具导出功能
"""

import json
import csv
import io
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from .palettes import get_palette


class PatternExporter:
    """图案导出器"""
    
    def __init__(self, palette_name: str = "perler"):
        """
        初始化导出器
        
        Args:
            palette_name: 调色板名称
        """
        self.palette_name = palette_name
        # 构建 {颜色名称: 十六进制} 映射，并统一加上 '#'
        palette_list = get_palette(palette_name) or []
        self.palette_colors = {
            c.get('name'): (c.get('color') if str(c.get('color','')).startswith('#') else f"#{c.get('color')}")
            for c in palette_list
        }
    
    def export_color_statistics(self, color_stats: Dict[str, int], format: str = "json") -> str:
        """
        导出颜色统计信息
        
        Args:
            color_stats: 颜色统计数据
            format: 导出格式 ("json", "csv", "txt")
            
        Returns:
            导出的字符串数据
        """
        if format == "json":
            return self._export_json_stats(color_stats)
        elif format == "csv":
            return self._export_csv_stats(color_stats)
        elif format == "txt":
            return self._export_txt_stats(color_stats)
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    
    def _export_json_stats(self, color_stats: Dict[str, int]) -> str:
        """导出JSON格式的颜色统计"""
        export_data = {
            "palette": self.palette_name,
            "total_beads": sum(color_stats.values()),
            "unique_colors": len(color_stats),
            "colors": []
        }
        
        for color_name, count in color_stats.items():
            color_hex = self.palette_colors.get(color_name, "#000000")
            export_data["colors"].append({
                "name": color_name,
                "hex": color_hex,
                "count": count,
                "percentage": round(count / sum(color_stats.values()) * 100, 2)
            })
        
        return json.dumps(export_data, ensure_ascii=False, indent=2)
    
    def _export_csv_stats(self, color_stats: Dict[str, int]) -> str:
        """导出CSV格式的颜色统计"""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # 写入标题行
        writer.writerow(["颜色名称", "十六进制", "数量", "百分比"])
        
        total_beads = sum(color_stats.values())
        for color_name, count in color_stats.items():
            color_hex = self.palette_colors.get(color_name, "#000000")
            percentage = round(count / total_beads * 100, 2)
            writer.writerow([color_name, color_hex, count, f"{percentage}%"])
        
        return output.getvalue()
    
    def _export_txt_stats(self, color_stats: Dict[str, int]) -> str:
        """导出文本格式的颜色统计"""
        lines = [
            f"拼豆图案颜色统计 - {self.palette_name.upper()}",
            "=" * 50,
            f"总拼豆数: {sum(color_stats.values())}",
            f"颜色种类: {len(color_stats)}",
            "",
            "颜色详情:",
            "-" * 30
        ]
        
        total_beads = sum(color_stats.values())
        for i, (color_name, count) in enumerate(color_stats.items(), 1):
            color_hex = self.palette_colors.get(color_name, "#000000")
            percentage = round(count / total_beads * 100, 2)
            lines.append(f"{i:2d}. {color_name:8s} {color_hex:8s} {count:6d} 个 ({percentage:5.1f}%)")
        
        return "\n".join(lines)
    
    def create_shopping_list(self, color_stats: Dict[str, int], 
                           bags_per_color: int = 1000) -> str:
        """
        创建购买清单
        
        Args:
            color_stats: 颜色统计数据
            bags_per_color: 每袋拼豆数量
            
        Returns:
            购买清单文本
        """
        lines = [
            f"拼豆购买清单 - {self.palette_name.upper()}",
            "=" * 50,
            f"每袋拼豆数量: {bags_per_color}",
            "",
            "需要购买的颜色:",
            "-" * 30
        ]
        
        total_bags = 0
        for color_name, count in color_stats.items():
            bags_needed = max(1, (count + bags_per_color - 1) // bags_per_color)  # 向上取整
            total_bags += bags_needed
            color_hex = self.palette_colors.get(color_name, "#000000")
            lines.append(f"{color_name:8s} {color_hex:8s} - {bags_needed:2d} 袋 (需要 {count:4d} 个)")
        
        lines.extend([
            "",
            f"总计需要: {total_bags} 袋",
            f"总拼豆数: {sum(color_stats.values())} 个"
        ])
        
        return "\n".join(lines)
    
    def create_pattern_guide(self, image: np.ndarray, piece_size: int = 20) -> Image.Image:
        """
        创建图案指南图片
        
        Args:
            image: 处理后的图像数组
            piece_size: 拼豆尺寸
            
        Returns:
            PIL图像对象
        """
        height, width = image.shape[:2]
        bead_width = width // piece_size
        bead_height = height // piece_size
        
        # 创建指南图像 (放大以便显示文字)
        guide_width = bead_width * 60  # 每个拼豆60像素
        guide_height = bead_height * 60
        
        guide_img = Image.new('RGB', (guide_width, guide_height), 'white')
        draw = ImageDraw.Draw(guide_img)
        
        try:
            # 尝试使用系统字体
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            # 如果没有找到字体，使用默认字体
            font = ImageFont.load_default()
        
        # 绘制每个拼豆位置和颜色名称
        for y in range(bead_height):
            for x in range(bead_width):
                # 获取该位置的颜色
                pixel_x = x * piece_size + piece_size // 2
                pixel_y = y * piece_size + piece_size // 2
                
                if pixel_y < height and pixel_x < width:
                    pixel_color = image[pixel_y, pixel_x]
                    
                    # 找到最接近的调色板颜色
                    from .color_utils import ColorUtils
                    color_utils = ColorUtils()
                    closest = color_utils.find_closest_palette_color(
                        tuple(pixel_color[:3]), self.palette_name
                    )
                    
                    # 绘制拼豆区域
                    left = x * 60
                    top = y * 60
                    right = left + 60
                    bottom = top + 60
                    
                    # 填充颜色
                    # 使用最接近颜色的十六进制（补全 '#')
                    color_hex = closest.get('color', '#000000')
                    if not str(color_hex).startswith('#'):
                        color_hex = f"#{color_hex}"
                    draw.rectangle([left, top, right, bottom], fill=color_hex, outline='black')
                    
                    # 添加颜色名称
                    color_name = closest.get('name', '')
                    text_color = 'white' if self._is_dark_color(color_hex) else 'black'
                    text_bbox = draw.textbbox((0, 0), color_name, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    
                    text_x = left + (60 - text_width) // 2
                    text_y = top + (60 - text_height) // 2
                    
                    draw.text((text_x, text_y), color_name, fill=text_color, font=font)
        
        return guide_img
    
    def _is_dark_color(self, hex_color: str) -> bool:
        """判断颜色是否为深色"""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        # 使用亮度公式
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        return brightness < 128
    
    def export_pattern_data(self, image: np.ndarray, piece_size: int = 20) -> Dict:
        """
        导出完整的图案数据
        
        Args:
            image: 处理后的图像数组
            piece_size: 拼豆尺寸
            
        Returns:
            包含图案信息的字典
        """
        height, width = image.shape[:2]
        bead_width = width // piece_size
        bead_height = height // piece_size
        
        pattern_data = {
            "metadata": {
                "palette": self.palette_name,
                "bead_size": piece_size,
                "pattern_width": bead_width,
                "pattern_height": bead_height,
                "total_beads": bead_width * bead_height
            },
            "pattern": []
        }
        
        from .color_utils import ColorUtils
        color_utils = ColorUtils()
        
        # 生成图案数据
        for y in range(bead_height):
            row = []
            for x in range(bead_width):
                pixel_x = x * piece_size + piece_size // 2
                pixel_y = y * piece_size + piece_size // 2
                
                if pixel_y < height and pixel_x < width:
                    pixel_color = image[pixel_y, pixel_x]
                    closest = color_utils.find_closest_palette_color(
                        tuple(pixel_color[:3]), self.palette_name
                    )
                    row.append(closest.get('name'))
                else:
                    row.append(None)
            
            pattern_data["pattern"].append(row)
        
        return pattern_data