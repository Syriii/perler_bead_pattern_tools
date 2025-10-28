"""
图像切片处理模块
实现图像分割、坐标映射和切片管理功能
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional
import math
 


class ImageSlicer:
    """图像切片处理类"""
    
    def __init__(self, piece_size: int = 20):
        """
        初始化切片器
        
        Args:
            piece_size: 每个拼豆的像素大小
        """
        self.piece_size = piece_size
        self.slice_images = []
        self.slice_info = []  # 存储每个切片的信息
        self.grid_size = (0, 0)  # 网格尺寸 (cols, rows)
        
        
    def slice_image(self, image: Image.Image, max_slice_size: int = 400) -> List[Dict]:
        """
        将图像切片为小块
        
        Args:
            image: 要切片的图像
            max_slice_size: 每个切片的最大尺寸（像素）
            
        Returns:
            切片信息列表
        """
        if image is None:
            return []
        
        try:
            width, height = image.size
            
            # 计算每个切片包含的拼豆数量
            beads_per_slice = max_slice_size // self.piece_size
            
            # 计算网格尺寸（以拼豆为单位）
            beads_width = width // self.piece_size
            beads_height = height // self.piece_size
            
            # 计算切片数量
            slice_cols = math.ceil(beads_width / beads_per_slice)
            slice_rows = math.ceil(beads_height / beads_per_slice)
            
            self.grid_size = (slice_cols, slice_rows)
            self.slice_images = []
            self.slice_info = []
            
            # 生成切片
            for row in range(slice_rows):
                for col in range(slice_cols):
                    # 计算切片在拼豆网格中的位置
                    start_bead_x = col * beads_per_slice
                    start_bead_y = row * beads_per_slice
                    end_bead_x = min(start_bead_x + beads_per_slice, beads_width)
                    end_bead_y = min(start_bead_y + beads_per_slice, beads_height)
                    
                    # 转换为像素坐标
                    start_pixel_x = start_bead_x * self.piece_size
                    start_pixel_y = start_bead_y * self.piece_size
                    end_pixel_x = end_bead_x * self.piece_size
                    end_pixel_y = end_bead_y * self.piece_size
                    
                    # 裁剪图像
                    slice_image = image.crop((start_pixel_x, start_pixel_y, end_pixel_x, end_pixel_y))
                    
                    # 创建切片信息
                    slice_info = {
                        'index': len(self.slice_images),
                        'row': row,
                        'col': col,
                        'bead_start': (start_bead_x, start_bead_y),
                        'bead_end': (end_bead_x, end_bead_y),
                        'pixel_start': (start_pixel_x, start_pixel_y),
                        'pixel_end': (end_pixel_x, end_pixel_y),
                        'bead_size': (end_bead_x - start_bead_x, end_bead_y - start_bead_y),
                        'image': slice_image
                    }
                    
                    self.slice_images.append(slice_image)
                    self.slice_info.append(slice_info)
            
            return self.slice_info
            
        except Exception as e:
            print(f"图像切片失败: {e}")
            return []
    
    def create_mosaic_view(self, image: Image.Image, palette_name: str, 
                          show_grid: bool = True, show_rulers: bool = True) -> Image.Image:
        """
        创建马赛克视图，显示拼豆效果
        
        Args:
            image: 量化后的图像
            palette_name: 调色板名称
            show_grid: 是否显示网格
            show_rulers: 是否显示标尺
            
        Returns:
            马赛克视图图像
        """
        if image is None:
            return None
        
        try:
            width, height = image.size
            
            # 计算拼豆网格尺寸
            beads_width = width // self.piece_size
            beads_height = height // self.piece_size
            
            # 创建输出图像（包含标尺空间）
            ruler_size = 30 if show_rulers else 0
            output_width = width + ruler_size
            output_height = height + ruler_size
            
            output_image = Image.new('RGBA', (output_width, output_height), (255, 255, 255, 255))
            draw = ImageDraw.Draw(output_image)
            
            # 绘制图像内容
            img_array = np.array(image)
            
            # 构建RGB->名称映射（量化后像素应为调色板颜色）
            from .color_utils import ColorUtils
            rgb_to_name = {tuple(v): k for k, v in ColorUtils().generate_palette_map(palette_name).items()}

            for bead_y in range(beads_height):
                for bead_x in range(beads_width):
                    # 计算像素位置
                    pixel_x = bead_x * self.piece_size
                    pixel_y = bead_y * self.piece_size
                    
                    # 获取该区域的主要颜色（多数票）
                    region = img_array[pixel_y:pixel_y + self.piece_size, 
                                     pixel_x:pixel_x + self.piece_size]
                    
                    if region.size > 0:
                        counts_local = {}
                        for ry in range(region.shape[0]):
                            for rx in range(region.shape[1]):
                                pixel = region[ry, rx]
                                if pixel[3] == 0:
                                    continue
                                rgb = tuple(pixel[:3])
                                name = rgb_to_name.get(rgb)
                                if name:
                                    counts_local[name] = counts_local.get(name, 0) + 1

                        # 默认用区域平均色作为回退（极端情况下不在调色板）
                        fill_rgb = None
                        if counts_local:
                            dominant_name = max(counts_local.items(), key=lambda kv: kv[1])[0]
                            # 通过palette映射拿回RGB
                            fill_rgb = ColorUtils().generate_palette_map(palette_name)[dominant_name]
                        else:
                            avg_color = np.mean(region.reshape(-1, region.shape[-1]), axis=0)
                            fill_rgb = tuple(avg_color[:3].astype(int))

                        # 绘制拼豆
                        bead_rect = (
                            ruler_size + pixel_x,
                            ruler_size + pixel_y,
                            ruler_size + pixel_x + self.piece_size,
                            ruler_size + pixel_y + self.piece_size
                        )
                        
                        draw.rectangle(bead_rect, fill=tuple(fill_rgb))
                        
                        # 绘制网格
                        if show_grid:
                            draw.rectangle(bead_rect, outline=(128, 128, 128), width=1)
            
            # 绘制标尺
            if show_rulers:
                self._draw_rulers(draw, beads_width, beads_height, ruler_size)
            
            return output_image
            
        except Exception as e:
            print(f"创建马赛克视图失败: {e}")
            return None
    
    def _draw_rulers(self, draw: ImageDraw.Draw, beads_width: int, beads_height: int, ruler_size: int):
        """
        绘制标尺
        
        Args:
            draw: 绘图对象
            beads_width: 拼豆宽度
            beads_height: 拼豆高度
            ruler_size: 标尺尺寸
        """
        try:
            # 绘制水平标尺
            for i in range(0, beads_width + 1, 5):
                x = ruler_size + i * self.piece_size
                
                # 绘制刻度线
                if i % 10 == 0:
                    draw.line([(x, 0), (x, ruler_size)], fill=(0, 0, 0), width=2)
                    # 绘制数字
                    if i > 0:
                        draw.text((x - 10, 5), str(i), fill=(0, 0, 0))
                else:
                    draw.line([(x, ruler_size - 10), (x, ruler_size)], fill=(0, 0, 0), width=1)
            
            # 绘制垂直标尺
            for i in range(0, beads_height + 1, 5):
                y = ruler_size + i * self.piece_size
                
                # 绘制刻度线
                if i % 10 == 0:
                    draw.line([(0, y), (ruler_size, y)], fill=(0, 0, 0), width=2)
                    # 绘制数字
                    if i > 0:
                        draw.text((5, y - 10), str(i), fill=(0, 0, 0))
                else:
                    draw.line([(ruler_size - 10, y), (ruler_size, y)], fill=(0, 0, 0), width=1)
                    
        except Exception as e:
            print(f"绘制标尺失败: {e}")
    
    def get_total_slices(self) -> int:
        """获取切片总数"""
        return len(self.slice_images)