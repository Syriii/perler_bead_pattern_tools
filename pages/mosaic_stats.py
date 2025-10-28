"""
马赛克统计页面 - 多页面应用
上传拼豆图纸，统计每格颜色对应的拼豆编码及数量
"""

import streamlit as st
import numpy as np
from PIL import Image

from tools import (
    ColorUtils,
    get_palette_names,
    get_palette_colors,
)

# 多页面应用中，页面配置在主 app.py 设置，避免重复调用


    


def main():
    st.title("🧮 上传拼豆图纸统计")
    st.markdown("上传由本工具或其他来源生成的拼豆图纸，设置单格像素大小与可选标尺尺寸，统计每种拼豆颜色的格子数量。")

    # 左侧标签栏：放置全局设置（跨页面共享）
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

    # 上传与参数设置
    mosaic_file = st.file_uploader("选择拼豆图纸图片", type=['png', 'jpg', 'jpeg', 'bmp'], key="mosaic_uploader")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        cell_size = st.number_input(
            "单格像素尺寸",
            min_value=5,
            max_value=400,
            value=60,
            step=5,
            help="每个马赛克格子的像素边长；本工具导出的图案指南为60。",
        )
    with col_m2:
        ruler_size_mosaic = st.number_input(
            "标尺尺寸（像素）",
            min_value=0,
            max_value=200,
            value=0,
            step=5,
            help="如果图片左上有标尺边距，请填写其像素宽度；无则为0。",
        )

    if mosaic_file is not None:
        mosaic_img = Image.open(mosaic_file).convert('RGBA')
        st.image(mosaic_img, caption="拼豆图纸（原图）", use_container_width=True)

        if st.button("统计马赛克颜色格数", type="primary", key="mosaic_stat_btn"):
            try:
                img_array = np.array(mosaic_img)
                height, width = img_array.shape[:2]

                # 去除标尺区域（默认在左与上）
                start_x = int(ruler_size_mosaic)
                start_y = int(ruler_size_mosaic)
                usable_w = max(0, width - start_x)
                usable_h = max(0, height - start_y)

                beads_w = usable_w // int(cell_size)
                beads_h = usable_h // int(cell_size)
                if beads_w <= 0 or beads_h <= 0:
                    st.error("参数可能不正确：请检查单格像素尺寸与标尺尺寸。")
                else:
                    # 构建调色板映射与反向映射
                    color_utils = ColorUtils()
                    palette_map = color_utils.generate_palette_map(selected_palette)
                    rgb_to_name = {tuple(rgb): name for name, rgb in palette_map.items()}

                    counts_global = {}
                    for by in range(beads_h):
                        for bx in range(beads_w):
                            px = start_x + bx * int(cell_size)
                            py = start_y + by * int(cell_size)
                            region = img_array[py:py + int(cell_size), px:px + int(cell_size)]
                            if region.size == 0:
                                continue
                            # 多数票统计（优先匹配到调色板RGB），否则用区域平均色映射到最近颜色
                            local_counts = {}
                            for ry in range(region.shape[0]):
                                for rx in range(region.shape[1]):
                                    pixel = region[ry, rx]
                                    if pixel[3] == 0:
                                        continue
                                    rgb = tuple(pixel[:3])
                                    name = rgb_to_name.get(rgb)
                                    if name:
                                        local_counts[name] = local_counts.get(name, 0) + 1

                            if local_counts:
                                dominant_name = max(local_counts.items(), key=lambda kv: kv[1])[0]
                            else:
                                avg_color = np.mean(region.reshape(-1, region.shape[-1]), axis=0)
                                closest = color_utils.find_closest_palette_color(tuple(avg_color[:3].astype(int)), selected_palette)
                                dominant_name = closest['name']

                            counts_global[dominant_name] = counts_global.get(dominant_name, 0) + 1

                    # 渲染统计表（编码+色块同列，仅显示前20项）
                    palette_colors = get_palette_colors(selected_palette)
                    total_cells = sum(counts_global.values())
                    rows_html = []
                    for name, cnt in sorted(counts_global.items(), key=lambda kv: kv[1], reverse=True)[:20]:
                        hex_color = ColorUtils.normalize_hex(palette_colors.get(name, '#FFFFFF'))
                        cell = (
                            f"<div style='display:flex;align-items:center;gap:8px'>"
                            f"<span style='display:inline-block;width:24px;height:24px;border:1px solid #ccc;background-color:{hex_color}'></span>"
                            f"<code>{name}</code>"
                            f"</div>"
                        )
                        pct = f"{round(cnt / total_cells * 100, 2)}%" if total_cells else "0%"
                        rows_html.append(f"<tr><td>{cell}</td><td>{cnt}</td><td>{pct}</td></tr>")

                    table_html = (
                        "<table style='width:100%;border-collapse:collapse'>"
                        "<thead><tr>"
                        "<th style='text-align:left;padding:6px;border-bottom:1px solid #ddd'>颜色编码</th>"
                        "<th style='text-align:left;padding:6px;border-bottom:1px solid #ddd'>数量</th>"
                        "<th style='text-align:left;padding:6px;border-bottom:1px solid #ddd'>占比</th>"
                        "</tr></thead>"
                        f"<tbody>{''.join(rows_html)}</tbody>"
                        "</table>"
                    )
                    st.markdown(table_html, unsafe_allow_html=True)

                    st.success(f"统计完成：总格数 {total_cells}，调色板 {selected_palette}")
            except Exception as e:
                st.error(f"统计失败：{e}")


if __name__ == "__main__":
    main()