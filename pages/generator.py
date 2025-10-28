"""
生成拼豆图案页面 - 多页面应用
将图片转换为拼豆图案并导出结果
"""

import streamlit as st
import numpy as np
from PIL import Image
import io

from tools import (
    ImageProcessor,
    ImageSlicer,
    ColorUtils,
    PatternExporter,
    get_palette_names,
    get_palette_colors,
)

# 初始化会话状态（本页使用到的键）
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'slicer' not in st.session_state:
    st.session_state.slicer = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'quantized_pil' not in st.session_state:
    st.session_state.quantized_pil = None
if 'quantized_array' not in st.session_state:
    st.session_state.quantized_array = None
if 'color_replacements' not in st.session_state:
    st.session_state.color_replacements = {}


def render_color_replacements_expander(selected_palette: str) -> None:
    """侧边栏：颜色替换设置区块"""
    with st.expander("颜色替换"):
        palette_colors_main = get_palette_colors(selected_palette)
        color_names_main = list(palette_colors_main.keys())
        if color_names_main:
            col_old, col_new = st.columns(2)
            with col_old:
                old_color_name = st.selectbox("原颜色", color_names_main, key="old_color_select")
                if old_color_name:
                    hex_old = palette_colors_main.get(old_color_name)
                    if hex_old:
                        st.markdown(
                            f"<div style='display:flex;align-items:center;gap:8px;margin:6px 0'>"
                            f"<span style='display:inline-block;width:24px;height:24px;border:1px solid #ccc;background-color:{ColorUtils.normalize_hex(hex_old)}'></span>"
                            f"<code>{old_color_name}</code>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
            with col_new:
                new_color_name = st.selectbox("新颜色", color_names_main, key="new_color_select")
                if new_color_name:
                    hex_new = palette_colors_main.get(new_color_name)
                    if hex_new:
                        st.markdown(
                            f"<div style='display:flex;align-items:center;gap:8px;margin:6px 0'>"
                            f"<span style='display:inline-block;width:24px;height:24px;border:1px solid #ccc;background-color:{ColorUtils.normalize_hex(hex_new)}'></span>"
                            f"<code>{new_color_name}</code>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
        # 间距
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if color_names_main and st.button("添加颜色替换规则"):
            st.session_state.color_replacements[old_color_name] = new_color_name
            st.success(f"已添加规则: {old_color_name} -> {new_color_name}")
        if st.session_state.color_replacements:
            st.caption("当前替换规则")
            table_rows = []
            for k, v in st.session_state.color_replacements.items():
                hex_old = ColorUtils.normalize_hex(palette_colors_main.get(k, '#FFFFFF'))
                hex_new = ColorUtils.normalize_hex(palette_colors_main.get(v, '#FFFFFF'))
                cell_old = (
                    f"<div style='display:flex;align-items:center;gap:8px'>"
                    f"<span style='display:inline-block;width:24px;height:24px;border:1px solid #ccc;background-color:{hex_old}'></span>"
                    f"<code>{k}</code>"
                    f"</div>"
                )
                cell_new = (
                    f"<div style='display:flex;align-items:center;gap:8px'>"
                    f"<span style='display:inline-block;width:24px;height:24px;border:1px solid #ccc;background-color:{hex_new}'></span>"
                    f"<code>{v}</code>"
                    f"</div>"
                )
                table_rows.append(f"<tr><td>{cell_old}</td><td>{cell_new}</td></tr>")
            table_html = (
                "<table style='width:100%;border-collapse:collapse'>"
                "<thead><tr>"
                "<th style='text-align:left;padding:6px;border-bottom:1px solid #ddd'>原颜色编码</th>"
                "<th style='text-align:left;padding:6px;border-bottom:1px solid #ddd'>新颜色编码</th>"
                "</tr></thead>"
                f"<tbody>{''.join(table_rows)}</tbody>"
                "</table>"
            )
            st.markdown(table_html, unsafe_allow_html=True)
            if st.button("清空替换规则"):
                st.session_state.color_replacements.clear()
                st.info("已清空替换规则")


def main():
    st.title("🎨 生成拼豆图案")
    st.markdown("上传图片并转换为拼豆图案！")

    # 左侧标签栏：放置全局设置
    with st.sidebar:
        st.header("全局设置")
        palette_names = get_palette_names()
        if 'selected_palette' not in st.session_state:
            st.session_state.selected_palette = "perler" if "perler" in palette_names else palette_names[0]
        selected_palette = st.selectbox(
            "拼豆品牌",
            palette_names,
            index=palette_names.index(st.session_state.selected_palette) if st.session_state.selected_palette in palette_names else 0,
            key="global_palette_select",
            help="选择拼豆品牌（全局设置）"
        )
        st.session_state.selected_palette = selected_palette

    # 侧边栏：拼豆图案设置
    with st.sidebar:
        st.divider()
        st.header("拼豆图案设置")
        piece_size = st.slider(
            "拼豆尺寸 (像素)", 5, 50, 20,
            help=(
                "定义每个拼豆单元的像素边长。值越小细节更丰富但拼豆更多；"
                "值越大制作更简便但细节减少。建议根据图片大小与期望细节在 10–30 调整。"
            ),
            key="piece_size_slider",
        )
        max_slice_size_beads = st.slider(
            "最大切片尺寸 (拼豆数)", 10, 100, 29,
            help=(
                "每个切片的最大边长（单位为拼豆数），用于将大图案分块以便打印或按板子规格拼装。"
                "常见 29×29 板建议设为 29；值越大切片更大数量更少，值越小切片更小数量更多。"
            ),
            key="max_slice_size_beads_slider",
        )
        with st.expander("高级选项"):
            trim_white_edges = st.checkbox("自动裁剪白色边缘", value=True, key="trim_white_edges_chk")
            trim_transparent_edges = st.checkbox("自动裁剪透明边缘", value=False, key="trim_transparent_edges_chk")
            show_grid = st.checkbox("显示网格", value=True, key="show_grid_chk")
            show_rulers = st.checkbox("显示标尺", value=True, key="show_rulers_chk")

        # 颜色替换（侧边栏设置项）
        render_color_replacements_expander(selected_palette)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 上传图片")
        uploaded_file = st.file_uploader(
            "选择图片文件",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            help="支持 PNG, JPG, JPEG, GIF, BMP 格式"
        )

        if uploaded_file is not None:
            # 加载图片
            image = Image.open(uploaded_file)
            st.session_state.original_image = image

            st.image(image, caption="原始图片", use_container_width=True)

            # 处理按钮
            if st.button("🔄 处理图片", type="primary"):
                with st.spinner("正在处理图片..."):
                    try:
                        # 创建图像处理器
                        processor = ImageProcessor(palette_name=selected_palette)
                        processor.set_piece_size(piece_size)

                        # 加载并处理图像
                        processor.load_image_from_array(np.array(image))
                        if trim_white_edges:
                            processor.trim_white_edges()
                        if trim_transparent_edges:
                            processor.trim_transparent_edges()

                        # 颜色量化
                        processor.quantize_to_palette()

                        # 应用颜色替换（如有）
                        if st.session_state.color_replacements:
                            for old_name, new_name in st.session_state.color_replacements.items():
                                processor.add_color_replacement(old_name, new_name)
                            processor.apply_color_replacements()

                        st.session_state.processor = processor
                        st.session_state.quantized_pil = processor.quantized_image
                        st.session_state.quantized_array = np.array(processor.quantized_image)

                        # 创建切片器
                        slicer = ImageSlicer(piece_size=piece_size)
                        slicer.slice_image(
                            st.session_state.quantized_pil,
                            max_slice_size= max_slice_size_beads * piece_size
                        )
                        st.session_state.slicer = slicer

                        st.success("图片处理完成！")

                    except Exception as e:
                        st.error(f"处理图片时出错: {str(e)}")

    with col2:
        st.header("📥 处理结果")

        if st.session_state.quantized_pil is not None:
            # 显示处理后的图片
            processed_pil = st.session_state.quantized_pil
            st.image(processed_pil, caption="处理后图片", use_container_width=True)

            # 创建拼豆图纸
            if st.session_state.slicer is not None:
                mosaic_pil = st.session_state.slicer.create_mosaic_view(
                    st.session_state.quantized_pil,
                    palette_name=selected_palette,
                    show_grid=show_grid,
                    show_rulers=show_rulers
                )
                st.image(mosaic_pil, caption="拼豆图纸", use_container_width=True)
        else:
            st.info("请先上传并处理图片")

    # 结果信息和操作
    if st.session_state.processor is not None:
        st.header("📊 图案信息")

        col1, col2, col3 = st.columns(3)

        with col1:
            w, h = st.session_state.quantized_pil.size
            st.metric("图片尺寸", f"{w} × {h}")

        with col2:
            bead_width = w // piece_size
            bead_height = h // piece_size
            st.metric("拼豆尺寸", f"{bead_width} × {bead_height}")

        with col3:
            if st.session_state.slicer is not None:
                slice_count = st.session_state.slicer.get_total_slices()
                st.metric("切片数量", slice_count)

        # 颜色统计
        st.subheader("🎨 颜色统计")
        color_stats = st.session_state.processor.get_color_statistics()

        if color_stats:
            # 创建颜色统计表格（编码+色块同列，显示前10种颜色）
            palette_colors = get_palette_colors(selected_palette)
            total = sum(color_stats.values())
            stats_rows = []
            for color_name, count in list(color_stats.items())[:10]:
                if color_name in palette_colors:
                    hex_color = ColorUtils.normalize_hex(palette_colors[color_name])
                    cell_color = (
                        f"<div style='display:flex;align-items:center;gap:8px'>"
                        f"<span style='display:inline-block;width:24px;height:24px;border:1px solid #ccc;background-color:{hex_color}'></span>"
                        f"<code>{color_name}</code>"
                        f"</div>"
                    )
                    percent = f"{count/total*100:.1f}%"
                    stats_rows.append(f"<tr><td>{cell_color}</td><td>{count}</td><td>{percent}</td></tr>")
            if stats_rows:
                table_html = (
                    "<table style='width:100%;border-collapse:collapse'>"
                    "<thead><tr>"
                    "<th style='text-align:left;padding:6px;border-bottom:1px solid #ddd'>颜色编码</th>"
                    "<th style='text-align:left;padding:6px;border-bottom:1px solid #ddd'>数量</th>"
                    "<th style='text-align:left;padding:6px;border-bottom:1px solid #ddd'>百分比</th>"
                    "</tr></thead>"
                    f"<tbody>{''.join(stats_rows)}</tbody>"
                    "</table>"
                )
                st.markdown(table_html, unsafe_allow_html=True)

        # 切片信息
        if st.session_state.slicer is not None:
            st.subheader("🧩 切片信息")
            if st.session_state.slicer.slice_info:
                slice_rows = []
                for i, info in enumerate(st.session_state.slicer.slice_info[:5]):
                    bx1, by1 = info['bead_start']
                    bx2, by2 = info['bead_end']
                    bw, bh = info['bead_size']
                    slice_rows.append({
                        "切片": f"#{i+1}",
                        "网格位置": f"({info['col']}, {info['row']})",
                        "拼豆范围": f"({bx1}, {by1}) - ({bx2}, {by2})",
                        "尺寸": f"{bw} × {bh}"
                    })
                st.dataframe(slice_rows, use_container_width=True)

        # 导出功能
        st.header("💾 导出")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("下载处理后图片") and st.session_state.quantized_pil is not None:
                img_buffer = io.BytesIO()
                st.session_state.quantized_pil.save(img_buffer, format='PNG')
                st.download_button(
                    label="📥 下载 PNG",
                    data=img_buffer.getvalue(),
                    file_name="perler_pattern.png",
                    mime="image/png"
                )

        with col2:
            if st.button("下载拼豆图纸") and st.session_state.slicer is not None:
                mosaic_pil = st.session_state.slicer.create_mosaic_view(
                    st.session_state.quantized_pil,
                    palette_name=selected_palette,
                    show_grid=True,
                    show_rulers=True,
                )
                mosaic_buffer = io.BytesIO()
                mosaic_pil.save(mosaic_buffer, format='PNG')
                st.download_button(
                    label="📥 下载拼豆图纸",
                    data=mosaic_buffer.getvalue(),
                    file_name="perler_sheet.png",
                    mime="image/png"
                )

        with col3:
            if st.button("下载颜色统计 (JSON)"):
                exporter = PatternExporter(selected_palette)
                stats_json = exporter.export_color_statistics(color_stats, format="json")
                st.download_button(
                    label="📥 下载统计 (JSON)",
                    data=stats_json,
                    file_name="color_statistics.json",
                    mime="application/json"
                )
            if st.button("下载颜色统计 (CSV)"):
                exporter = PatternExporter(selected_palette)
                stats_csv = exporter.export_color_statistics(color_stats, format="csv")
                st.download_button(
                    label="📥 下载统计 (CSV)",
                    data=stats_csv,
                    file_name="color_statistics.csv",
                    mime="text/csv"
                )
            if st.button("下载颜色统计 (TXT)"):
                exporter = PatternExporter(selected_palette)
                stats_txt = exporter.export_color_statistics(color_stats, format="txt")
                st.download_button(
                    label="📥 下载颜色统计 (TXT)",
                    data=stats_txt,
                    file_name="color_statistics.txt",
                    mime="text/plain"
                )

    # 其他导出
    st.subheader("🛍️ 购买清单与图案指南")
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        bags_per_color = st.number_input("每袋拼豆数量", min_value=100, max_value=5000, value=1000, step=50)
        if st.button("生成并下载购买清单"):
            exporter = PatternExporter(selected_palette)
            shopping_txt = exporter.create_shopping_list(color_stats, bags_per_color=bags_per_color)
            st.download_button(
                label="📥 下载购买清单",
                data=shopping_txt,
                file_name="shopping_list.txt",
                mime="text/plain"
            )
    with exp_col2:
        if st.button("生成并下载图案指南") and st.session_state.quantized_array is not None:
            exporter = PatternExporter(selected_palette)
            guide_img = exporter.create_pattern_guide(st.session_state.quantized_array, piece_size=piece_size)
            buf = io.BytesIO()
            guide_img.save(buf, format='PNG')
            st.download_button(
                label="📥 下载图案指南",
                data=buf.getvalue(),
                file_name="pattern_guide.png",
                mime="image/png"
            )


if __name__ == "__main__":
    main()