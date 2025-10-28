# 拼豆图案工具 (Perler Bead Pattern Tool)

一个功能强大的Python工具，用于将图片转换为拼豆图案。支持多种拼豆品牌，提供完整的图像处理、切片和导出功能。

## 功能特性

### 🎨 核心功能
- **图像处理**: 自动调整图像大小、裁剪边缘、颜色量化
- **多品牌支持**: 支持 Perler、Hama、Artkal、Nabbi 等多个拼豆品牌
- **智能切片**: 将大图案分割为可管理的小块
- **颜色替换**: 支持单像素和批量颜色替换
- **统计分析**: 详细的颜色使用统计和购买建议

### 🖥️ 用户界面
- **Web界面**: 基于 Streamlit 的现代化 Web 界面
- **实时预览**: 即时查看处理结果和马赛克效果
- **交互式操作**: 直观的参数调整和结果预览

### 📤 导出功能
- **多格式导出**: JSON、CSV、TXT 格式的颜色统计
- **购买清单**: 自动生成拼豆购买建议
- **图案指南**: 带有颜色标注的详细制作指南
- **完整数据**: 包含所有图案信息的结构化数据

## 安装与启动

### 环境要求
- Python 3.10+
- 依赖包见 `requirements.txt`

### 安装依赖
```bash
pip install -r requirements.txt
```

### 启动方法
```bash
streamlit run app.py
```
- 打开浏览器访问：`http://localhost:8501/`
- 如果端口被占用，可指定其他端口：
```bash
streamlit run app.py --server.port 8502
```

提示：本项目采用 Streamlit 多页面结构，主入口为欢迎页，`pages/generator.py` 为生成拼豆图案页面，`pages/mosaic_stats.py` 为马赛克颜色统计页面。可通过左侧页面导航或首页链接进入。

### 使用方法

1. **上传图片**: 支持 PNG、JPG、JPEG、GIF、BMP 格式
2. **选择设置**: 
   - 选择拼豆品牌（Perler、Hama 等）
   - 调整拼豆尺寸和图案大小
   - 设置切片参数
3. **处理图片**: 点击"处理图片"按钮
4. **查看结果**: 
   - 查看处理后的图案
- 查看拼豆图纸
   - 查看颜色统计和切片信息
5. **导出结果**: 下载图案、统计数据或购买清单

## 参数说明与建议

- 拼豆尺寸（像素）
  - 含义：定义每个拼豆单元的像素边长。该值影响网格粒度与最终所需拼豆数量。
  - 影响：值越小，网格更细、细节更丰富，但需要更多拼豆；值越大，网格更粗、细节更少，但更易制作与导出。
  - 建议：根据图片尺寸与期望细节调整，一般在 `10–30` 像素之间较为均衡。小图或追求细节选小值，大图或追求效率选大值。

- 最大切片尺寸（拼豆数）
  - 含义：每个切片的最大边长（单位为拼豆数），用于将大图案分块以便打印或按板子规格拼装。
  - 影响：值越大，单个切片更大、切片数量更少；值越小，单个切片更小、数量更多，便于普通打印机与小板子拼装。
  - 建议：若使用常见 `29×29` 拼豆板，推荐设为 `29`；如使用其它板型，请按板子边长设定。若需减少切片数量可适当增大，追求更灵活拼装则适当减小。

### 常见场景参数选择表

| 场景 | 目标 | 推荐拼豆尺寸（像素） | 推荐最大切片尺寸（拼豆数） | 适用板规格 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 小型作品（钥匙扣、胸章） | 快速制作、低用量 | 20–30 | 20–29 | 小板或 29×29 | 适度降低细节以提高效率 |
| 中型作品（桌面摆件、卡片） | 细节与效率平衡 | 15–25 | 29–40 | 29×29 或更大板 | 通用推荐：`20 / 29` |
| 大型作品（墙面海报） | 减少切片数量 | 20–35 | 40–80 | 大板拼组 | 优先效率，细节适度取舍 |
| 高清照片类图像 | 细节保真 | 10–18 | 29–48 | 29×29 或更大板 | 拼豆用量高、制作时间长 |
| 像素画 / 简笔画 / Logo | 线条清晰 | 15–25 | 29–40 | 29×29 | 中等粒度足够表现轮廓 |
| 受限打印设备（A4 单页） | 单页可打印 | 15–25 | 20–29 | 29×29 | 导出时选合适 DPI，避免溢出 |
| 固定板规格优先 | 匹配板子 | 15–25（随图） | 与板边长一致（如 29、30、48） | 对应板规格 | 切片边长=板边长便于拼装 |

说明：
- 拼豆尺寸（像素）范围支持 `5–50`，数值越小细节越丰富但用量和时间更高；数值越大制作更省力但细节减少。
- 最大切片尺寸（拼豆数）范围支持 `10–100`，请根据打印设备、板尺寸与拼装偏好选择。应用内会按 `切片尺寸 × 拼豆尺寸` 转换为像素用于切分。

## 项目结构

```
perler_bead_pattern_tools/
├── perler_bead_tool/           # 核心包
│   ├── __init__.py            # 包初始化
│   ├── palettes.py            # 调色板数据
│   ├── color_utils.py         # 颜色处理工具
│   ├── image_processor.py     # 图像处理器
│   ├── slicer.py             # 图像切片器
│   └── exporter.py           # 导出功能
├── app.py                     # Streamlit Web 应用
├── pages/                     # Streamlit 多页面目录
│   └── mosaic_stats.py        # 马赛克颜色统计页
│   └── generator.py           # 生成拼豆图案页
├── requirements.txt           # 依赖包列表
└── README.md                  # 项目说明
```

## 支持的拼豆品牌

- **Perler**: 89 种颜色
- **Hama**: 51 种颜色  
- **Artkal-S**: 多种颜色
- **Artkal-R**: 多种颜色
- **Artkal-A**: 多种颜色
- **Nabbi**: 多种颜色
- **Mard**: 188 种颜色

## 技术特点

### 颜色处理算法
- 使用加权欧几里得距离进行颜色匹配
- 支持颜色缓存以提高性能
- 智能颜色量化算法

### 图像处理
- 自动边缘检测和裁剪
- 智能图像缩放
- 支持透明度处理

### 切片算法
- 智能切片分割
- 坐标映射和转换
- 拼豆图纸生成

## API 使用示例

### 基本使用
```python
from tools import ImageProcessor, ImageSlicer, PatternExporter

# 创建图像处理器
processor = ImageProcessor(palette_name="perler")
processor.set_piece_size(20)

# 加载和处理图像（从数组或路径均可）
processor.load_image("path/to/image.jpg")
processor.trim_white_edges()      # 可选：裁剪白边
processor.trim_transparent_edges()# 可选：裁剪透明边
processor.quantize_to_palette()   # 颜色量化到选定调色板

# 获取颜色统计
color_stats = processor.get_color_statistics()

# 切片与拼豆图纸
slicer = ImageSlicer(piece_size=20)
slicer.slice_image(processor.quantized_image, max_slice_size=29*20)
mosaic = slicer.create_mosaic_view(processor.quantized_image, palette_name="perler")

# 导出颜色统计与购买清单
exporter = PatternExporter("perler")
json_stats = exporter.export_color_statistics(color_stats, "json")
shopping_list = exporter.create_shopping_list(color_stats)
```

### 高级功能
```python
# 颜色替换
processor.add_color_replacement("P04", "P01")             # 批量替换
processor.add_pixel_color_replacement(10, 15, "P05")       # 单像素替换
processor.apply_color_replacements()

# 创建图案指南
guide_image = exporter.create_pattern_guide(
    image=processor.get_quantized_image_array() if hasattr(processor, 'get_quantized_image_array') else np.array(processor.quantized_image),
    piece_size=20
)
guide_image.save("pattern_guide.png")

# 导出完整图案数据
pattern_data = exporter.export_pattern_data(np.array(processor.quantized_image), piece_size=20)
```

## 从 JavaScript 版本迁移

本项目是从原始的 JavaScript 版本重构而来，主要改进包括：

1. **更好的性能**: Python 的 NumPy 和 PIL 提供更高效的图像处理
2. **更强的扩展性**: 模块化设计便于功能扩展
3. **更丰富的导出**: 支持多种格式的数据导出
4. **现代化界面**: 基于 Streamlit 的响应式 Web 界面

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。