"""
图像处理核心类
实现图像加载、颜色量化、边缘裁剪、切片等核心功能
"""

import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple, List, Dict, Optional, Union
import cv2
from .color_utils import ColorUtils
from .palettes import get_all_brands


class ImageProcessor:
    """图像处理核心类"""
    
    def __init__(self, palette_name: str = "perler"):
        """
        初始化图像处理器
        
        Args:
            palette_name: 调色板名称，默认为 "perler"
        """
        self.palette_name = palette_name
        self.color_utils = ColorUtils()
        self.original_image = None
        self.processed_image = None
        self.quantized_image = None
        self.slice_images = []
        self.piece_size = 20  # 每个拼豆的像素大小
        self.image_size_limit = 2000  # 图像尺寸限制
        
        # 颜色替换规则
        self.pixel_color_replace_list = {}  # 单像素颜色替换
        self.color_replace_list = {}  # 颜色替换（批量）
        
    def load_image(self, image_path: str) -> bool:
        """
        加载图像文件
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            是否成功加载
        """
        try:
            # 加载图像
            image = Image.open(image_path)
            
            # 转换为RGBA模式
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            
            # 检查图像尺寸
            width, height = image.size
            if width > self.image_size_limit or height > self.image_size_limit:
                # 等比例缩放
                ratio = min(self.image_size_limit / width, self.image_size_limit / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            self.original_image = image
            self.processed_image = image.copy()
            
            return True
            
        except Exception as e:
            print(f"加载图像失败: {e}")
            return False
    
    def load_image_from_array(self, image_array: np.ndarray) -> bool:
        """
        从numpy数组加载图像
        
        Args:
            image_array: 图像数组
            
        Returns:
            是否成功加载
        """
        try:
            # 确保数组格式正确
            if len(image_array.shape) == 3:
                if image_array.shape[2] == 3:  # RGB
                    # 添加alpha通道
                    alpha = np.ones((image_array.shape[0], image_array.shape[1], 1), dtype=image_array.dtype) * 255
                    image_array = np.concatenate([image_array, alpha], axis=2)
                elif image_array.shape[2] != 4:  # 不是RGBA
                    raise ValueError("图像数组必须是RGB或RGBA格式")
            else:
                raise ValueError("图像数组必须是3维的")
            
            # 转换为PIL图像
            image = Image.fromarray(image_array.astype(np.uint8), 'RGBA')
            
            # 检查尺寸限制
            width, height = image.size
            if width > self.image_size_limit or height > self.image_size_limit:
                ratio = min(self.image_size_limit / width, self.image_size_limit / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            self.original_image = image
            self.processed_image = image.copy()
            
            return True
            
        except Exception as e:
            print(f"从数组加载图像失败: {e}")
            return False
    
    def trim_transparent_edges(self) -> bool:
        """
        裁剪透明边缘
        
        Returns:
            是否成功裁剪
        """
        if self.processed_image is None:
            return False
        
        try:
            # 获取图像数组
            img_array = np.array(self.processed_image)
            
            # 找到非透明像素的边界
            alpha_channel = img_array[:, :, 3]
            non_transparent = alpha_channel > 0
            
            # 找到包含非透明像素的最小矩形
            rows = np.any(non_transparent, axis=1)
            cols = np.any(non_transparent, axis=0)
            
            if not np.any(rows) or not np.any(cols):
                # 图像完全透明
                return False
            
            row_min, row_max = np.where(rows)[0][[0, -1]]
            col_min, col_max = np.where(cols)[0][[0, -1]]
            
            # 裁剪图像
            cropped_array = img_array[row_min:row_max+1, col_min:col_max+1]
            self.processed_image = Image.fromarray(cropped_array, 'RGBA')
            
            return True
            
        except Exception as e:
            print(f"裁剪透明边缘失败: {e}")
            return False
    
    def trim_white_edges(self, threshold: int = 240) -> bool:
        """
        裁剪白色边缘
        
        Args:
            threshold: 白色阈值，像素值大于此值被认为是白色
            
        Returns:
            是否成功裁剪
        """
        if self.processed_image is None:
            return False
        
        try:
            # 转换为RGB进行处理
            rgb_image = self.processed_image.convert('RGB')
            img_array = np.array(rgb_image)
            
            # 找到非白色像素
            non_white = np.any(img_array < threshold, axis=2)
            
            # 找到包含非白色像素的最小矩形
            rows = np.any(non_white, axis=1)
            cols = np.any(non_white, axis=0)
            
            if not np.any(rows) or not np.any(cols):
                # 图像完全是白色
                return False
            
            row_min, row_max = np.where(rows)[0][[0, -1]]
            col_min, col_max = np.where(cols)[0][[0, -1]]
            
            # 裁剪原始RGBA图像
            original_array = np.array(self.processed_image)
            cropped_array = original_array[row_min:row_max+1, col_min:col_max+1]
            self.processed_image = Image.fromarray(cropped_array, 'RGBA')
            
            return True
            
        except Exception as e:
            print(f"裁剪白色边缘失败: {e}")
            return False
    
    def quantize_to_palette(self) -> bool:
        """
        将图像量化到指定调色板
        
        Returns:
            是否成功量化
        """
        if self.processed_image is None:
            return False
        
        try:
            # 获取图像数组
            img_array = np.array(self.processed_image)
            height, width = img_array.shape[:2]
            
            # 创建量化后的图像数组
            quantized_array = np.zeros_like(img_array)
            
            # 逐像素进行颜色量化（按原JS逻辑，采用alpha预乘）
            for y in range(height):
                for x in range(width):
                    pixel = img_array[y, x]
                    a = int(pixel[3])
                    
                    # 跳过透明像素
                    if a == 0:
                        quantized_array[y, x] = pixel
                        continue
                    
                    # 依据透明度对RGB进行预乘，贴近原可视效果
                    r = int(pixel[0]) * a // 255
                    g = int(pixel[1]) * a // 255
                    b = int(pixel[2]) * a // 255
                    rgb = (r, g, b)
                    
                    # 查找最接近的调色板颜色
                    closest_color = self.color_utils.find_closest_palette_color(rgb, self.palette_name)
                    
                    # 设置量化后的颜色，保留原alpha
                    quantized_array[y, x] = (*closest_color['rgb'], a)
            
            # 创建量化后的图像
            self.quantized_image = Image.fromarray(quantized_array.astype(np.uint8), 'RGBA')
            
            return True
            
        except Exception as e:
            print(f"颜色量化失败: {e}")
            return False
    
    def apply_color_replacements(self) -> bool:
        """
        应用颜色替换规则
        
        Returns:
            是否成功应用
        """
        if self.quantized_image is None:
            return False
        
        try:
            img_array = np.array(self.quantized_image)
            height, width = img_array.shape[:2]
            
            # 应用单像素颜色替换
            for (x, y), new_color_name in self.pixel_color_replace_list.items():
                if 0 <= x < width and 0 <= y < height:
                    # 获取新颜色的RGB值
                    palette_map = self.color_utils.generate_palette_map(self.palette_name)
                    if new_color_name in palette_map:
                        new_rgb = palette_map[new_color_name]
                        img_array[y, x, :3] = new_rgb
            
            # 应用颜色替换（批量）
            for old_color_name, new_color_name in self.color_replace_list.items():
                palette_map = self.color_utils.generate_palette_map(self.palette_name)
                if old_color_name in palette_map and new_color_name in palette_map:
                    old_rgb = palette_map[old_color_name]
                    new_rgb = palette_map[new_color_name]
                    
                    # 替换所有匹配的像素
                    mask = np.all(img_array[:, :, :3] == old_rgb, axis=2)
                    img_array[mask, :3] = new_rgb
            
            # 更新图像
            self.quantized_image = Image.fromarray(img_array.astype(np.uint8), 'RGBA')
            
            return True
            
        except Exception as e:
            print(f"应用颜色替换失败: {e}")
            return False
    
    def get_color_statistics(self) -> Dict[str, int]:
        """
        获取颜色统计信息（按拼豆单元计数，而非像素计数）
        
        Returns:
            颜色名称到数量的映射（每个拼豆单元计为1）
        """
        if self.quantized_image is None:
            return {}
        
        try:
            img_array = np.array(self.quantized_image)
            color_counts = {}
            palette_map = self.color_utils.generate_palette_map(self.palette_name)
            
            # 创建RGB到颜色名称的反向映射
            rgb_to_name = {rgb: name for name, rgb in palette_map.items()}

            # 统计每个拼豆单元的主色（多数票），每个单元计为1
            height, width = img_array.shape[:2]
            piece = max(1, getattr(self, 'piece_size', 1))
            beads_width = width // piece
            beads_height = height // piece

            for by in range(beads_height):
                for bx in range(beads_width):
                    px = bx * piece
                    py = by * piece
                    region = img_array[py:py + piece, px:px + piece]
                    if region.size == 0:
                        continue

                    # 统计该区域内每种调色板颜色的像素数
                    counts_local: Dict[str, int] = {}
                    for ry in range(region.shape[0]):
                        for rx in range(region.shape[1]):
                            pixel = region[ry, rx]
                            if pixel[3] == 0:
                                continue
                            rgb = tuple(pixel[:3])
                            name = rgb_to_name.get(rgb)
                            if name:
                                counts_local[name] = counts_local.get(name, 0) + 1

                    if counts_local:
                        # 选择多数票作为该拼豆单元的主色
                        dominant_name = max(counts_local.items(), key=lambda kv: kv[1])[0]
                        color_counts[dominant_name] = color_counts.get(dominant_name, 0) + 1
            
            return color_counts
            
        except Exception as e:
            print(f"获取颜色统计失败: {e}")
            return {}
    
    def set_palette(self, palette_name: str) -> bool:
        """
        设置调色板
        
        Args:
            palette_name: 调色板名称
            
        Returns:
            是否成功设置
        """
        if palette_name not in get_all_brands():
            return False
        
        self.palette_name = palette_name
        self.color_utils.clear_cache()  # 清空缓存
        
        return True
    
    def set_piece_size(self, piece_size: int):
        """设置拼豆尺寸"""
        self.piece_size = max(1, piece_size)
    
    def add_pixel_color_replacement(self, x: int, y: int, new_color_name: str):
        """添加单像素颜色替换规则"""
        self.pixel_color_replace_list[(x, y)] = new_color_name
    
    def add_color_replacement(self, old_color_name: str, new_color_name: str):
        """添加颜色替换规则（批量）"""
        self.color_replace_list[old_color_name] = new_color_name
    
    def clear_color_replacements(self):
        """清空所有颜色替换规则"""
        self.pixel_color_replace_list.clear()
        self.color_replace_list.clear()
    
    def get_image_size(self) -> Optional[Tuple[int, int]]:
        """获取当前处理图像的尺寸"""
        if self.processed_image is None:
            return None
        return self.processed_image.size