"""
Perler Bead Pattern Tool - Python版本

一个用于创建拼豆图案的工具，支持图像处理、颜色量化、切片和导出功能。
"""

__version__ = "1.0.0"
__author__ = "Perler Bead Tool Team"

from .palettes import ALL_PALETTES, get_palette, get_all_brands
from .image_processor import ImageProcessor
from .color_utils import ColorUtils
from .slicer import ImageSlicer
from .exporter import PatternExporter

# 添加兼容性函数
def get_palette_names():
    """获取所有调色板名称"""
    return get_all_brands()

def get_palette_colors(palette_name):
    """获取指定调色板的颜色"""
    palette = get_palette(palette_name)
    return {color['name']: color['color'] for color in palette}

__all__ = [
    'ALL_PALETTES',
    'get_palette', 
    'get_all_brands',
    'get_palette_names',
    'get_palette_colors',
    'ImageProcessor',
    'ColorUtils',
    'ImageSlicer',
    'PatternExporter'
]