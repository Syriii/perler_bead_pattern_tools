"""
网格线识别页面
仅保留网格线自动识别与预览相关逻辑
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

from tools import (
    get_palette_names,
    ColorUtils,
    get_palette_colors,
)

# 多页面应用中，页面配置在主 app.py 设置，避免重复调用





def _detect_grid_lines(
    img_rgba: np.ndarray,
    min_pitch: int = 10,
    max_pitch: int = 200,
    std_factor: float = 2.0,
    dilate_iters: int = 1,
    k_extra: int = 0,
    tol_factor: float = 0.25,
):
    """检测浅色/非纯黑网格线：自适应阈值 + 形态学提取 + 投影峰。

    更鲁棒地识别灰色/半透明网格线，并返回线坐标与格距。
    """
    try:
        h, w = img_rgba.shape[:2]
        # RGBA -> BGR -> GRAY
        bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 提升对比度（适应浅色线）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 去噪
        gray = cv2.medianBlur(gray, 3)

        # 自适应阈值，网格线变为白色（适配浅色线）
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21, 2
        )

        # 形态学提取垂直/水平线，并做轻微膨胀以连接断线
        k = max(3, min_pitch // 3) + int(k_extra)
        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))

        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ver_kernel, iterations=1)
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hor_kernel, iterations=1)
        # 适度膨胀，填补小间断（可调）
        vertical = cv2.dilate(vertical, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=max(0, int(dilate_iters)))
        horizontal = cv2.dilate(horizontal, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)), iterations=max(0, int(dilate_iters)))

        # 通过投影获取线位置（动态阈值，适配残缺线）
        col_sum = (vertical // 255).sum(axis=0)
        row_sum = (horizontal // 255).sum(axis=1)
        thr_v = max(5, int(np.mean(col_sum) + float(std_factor) * np.std(col_sum)))
        thr_h = max(5, int(np.mean(row_sum) + float(std_factor) * np.std(row_sum)))
        v_positions = [int(i) for i, v in enumerate(col_sum) if v >= thr_v]
        h_positions = [int(i) for i, v in enumerate(row_sum) if v >= thr_h]

        def _merge_positions(positions: list[int], tol: int = 2) -> list[int]:
            if not positions:
                return []
            positions = sorted(positions)
            merged = [positions[0]]
            for p in positions[1:]:
                if abs(p - merged[-1]) <= tol:
                    merged[-1] = (merged[-1] + p) // 2
                else:
                    merged.append(p)
            return merged

        v_positions = _merge_positions(v_positions, tol=2)
        h_positions = _merge_positions(h_positions, tol=2)

        def _estimate_pitch(positions: list[int]) -> int:
            if len(positions) < 3:
                return 0
            diffs = np.diff(positions)
            diffs = [d for d in diffs if min_pitch <= d <= max_pitch]
            if not diffs:
                return 0
            vals, counts = np.unique(diffs, return_counts=True)
            return int(vals[np.argmax(counts)]) if len(vals) else int(np.median(diffs))

        pitch_w = _estimate_pitch(v_positions)
        pitch_h = _estimate_pitch(h_positions)

        # 如果已得到稳定的间距，按间距补全缺失线位
        def _complete_grid_positions(positions: list[int], pitch: int, limit: int) -> list[int]:
            if pitch <= 0 or not positions:
                return positions
            positions = sorted(positions)
            # 估计起点为最靠近0的线位置（向左回推多个pitch）
            start = positions[0]
            while start - pitch >= 0:
                start -= pitch
            seq = []
            x = start
            while x <= limit:
                seq.append(int(round(x)))
                x += pitch
            # 将预测序列与检测结果对齐（容差为 pitch/4）
            tol = max(2, int(pitch * float(tol_factor)))
            merged = []
            for sx in seq:
                nearest = None
                if positions:
                    nearest = min(positions, key=lambda p: abs(p - sx))
                if nearest is not None and abs(nearest - sx) <= tol:
                    merged.append(int((nearest + sx) // 2))
                else:
                    merged.append(int(sx))
            return merged

        if pitch_w:
            v_positions = _complete_grid_positions(v_positions, pitch_w, w - 1)
        if pitch_h:
            h_positions = _complete_grid_positions(h_positions, pitch_h, h - 1)

        ok = pitch_w >= min_pitch and pitch_h >= min_pitch and len(v_positions) >= 3 and len(h_positions) >= 3

        # 构造预览图（叠加红/蓝）
        pil_preview = Image.fromarray(img_rgba.astype(np.uint8), 'RGBA').convert('RGB')
        draw = ImageDraw.Draw(pil_preview)
        for x in v_positions:
            draw.line([(x, 0), (x, h)], fill=(255, 0, 0), width=1)
        for y in h_positions:
            draw.line([(0, y), (w, y)], fill=(0, 128, 255), width=1)

        return {
            'ok': ok,
            'v_lines': v_positions,
            'h_lines': h_positions,
            'pitch_w': pitch_w,
            'pitch_h': pitch_h,
            'preview': pil_preview
        }
    except Exception:
        return {'ok': False, 'v_lines': [], 'h_lines': [], 'pitch_w': 0, 'pitch_h': 0, 'preview': None}


def _auto_detect_grid_lines(img_rgba: np.ndarray):
    """自动模式：无需用户调参，内部搜索多组参数并评分选优。

    步骤：
    1) 粗估格距（通过阈值后的投影自相关），给出[min_pitch, max_pitch]建议窗口。
    2) 遍历一组参数组合（std_factor, dilate_iters, k_extra, tol_factor），调用检测函数。
    3) 对结果按线数、间距均匀性（diff的标准差）、是否ok等综合评分，选择最佳。
    """
    try:
        h, w = img_rgba.shape[:2]
        # 灰度与二值（用于粗估格距）
        bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.medianBlur(gray, 3)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21, 2
        )
        col_sum = (binary // 255).sum(axis=0).astype(float)
        row_sum = (binary // 255).sum(axis=1).astype(float)

        def _rough_pitch_from_projection(proj: np.ndarray, min_lag: int = 8, max_lag: int = 128) -> int:
            if proj.size < max_lag + 5:
                max_lag = max(16, proj.size // 4)
            x = proj - proj.mean()
            ac = np.correlate(x, x, mode='full')
            ac = ac[ac.size // 2:]
            start = max(8, min_lag)
            end = min(len(ac) - 1, max_lag)
            if end <= start:
                return 0
            lag = int(np.argmax(ac[start:end]) + start)
            return lag if lag >= start else 0

        def _rough_pitch_fft(proj: np.ndarray, min_pitch: int = 8, max_pitch: int = 128) -> int:
            # 使用功率谱峰值估计周期，更稳健于噪声
            n = len(proj)
            if n < 16:
                return 0
            x = proj - proj.mean()
            # 汉宁窗减少泄漏
            win = np.hanning(n)
            xw = x * win
            fft = np.fft.rfft(xw)
            power = np.abs(fft) ** 2
            freqs = np.fft.rfftfreq(n, d=1.0)
            # 将频率映射为周期，并限制到期望窗口
            periods = np.zeros_like(freqs)
            with np.errstate(divide='ignore', invalid='ignore'):
                periods = np.where(freqs > 0, n / (freqs * n), 0)  # 近似周期（采样间距为1）
            mask = (periods >= min_pitch) & (periods <= max_pitch)
            if not np.any(mask):
                return 0
            idx = int(np.argmax(power[mask]))
            # 找到mask中的相对索引后转为绝对索引
            abs_idx = np.arange(len(power))[mask][idx]
            period = periods[abs_idx]
            return int(round(period)) if period > 0 else 0

        rough_w_ac = _rough_pitch_from_projection(col_sum)
        rough_h_ac = _rough_pitch_from_projection(row_sum)
        rough_w_fft = _rough_pitch_fft(col_sum)
        rough_h_fft = _rough_pitch_fft(row_sum)
        # 融合估计：优先选择两者一致的值，否则取非零的中位数
        candidates_w = [p for p in [rough_w_ac, rough_w_fft] if p and p >= 8]
        candidates_h = [p for p in [rough_h_ac, rough_h_fft] if p and p >= 8]
        rough_w = int(np.median(candidates_w)) if candidates_w else 0
        rough_h = int(np.median(candidates_h)) if candidates_h else 0
        # 建议窗口（若失败用宽泛默认）
        def _window(rough: int):
            if rough and rough >= 8:
                mn = max(5, int(round(rough * 0.8)))
                mx = int(round(rough * 1.2))
                return mn, mx
            return 10, 200

        min_pitch_w, max_pitch_w = _window(rough_w)
        min_pitch_h, max_pitch_h = _window(rough_h)
        # 统一窗口（取交集的范围，保持宽松）
        min_pitch = max(10, min(min_pitch_w, min_pitch_h))
        max_pitch = min(200, max(max_pitch_w, max_pitch_h))

        # 参数网格
        std_factors = [1.0, 1.3, 1.6, 2.0]
        dilates = [1, 2]
        k_extras = [0, 2, 3]
        tol_factors = [0.25, 0.30, 0.35]

        best = None
        best_score = -1e18

        def _score(res):
            v = res.get('v_lines', [])
            hls = res.get('h_lines', [])
            if len(v) < 3 or len(hls) < 3:
                return -1e9
            dv = np.diff(v)
            dh = np.diff(hls)
            # 使用MAD更鲁棒
            mad_v = float(np.median(np.abs(dv - np.median(dv)))) if dv.size else 1e6
            mad_h = float(np.median(np.abs(dh - np.median(dh)))) if dh.size else 1e6
            cnt = len(v) + len(hls)
            ok_bonus = 20 if res.get('ok') else 0
            pitch_bonus = 10 if (res.get('pitch_w', 0) > 0 and res.get('pitch_h', 0) > 0) else 0
            # 额外奖励：间距接近粗估
            pw, ph = res.get('pitch_w', 0), res.get('pitch_h', 0)
            close_bonus = 0
            if rough_w and pw:
                close_bonus += max(0, 10 - abs(pw - rough_w))
            if rough_h and ph:
                close_bonus += max(0, 10 - abs(ph - rough_h))
            # 更多线、间距更均匀更好
            return ok_bonus + pitch_bonus + close_bonus + cnt - 3.0 * (mad_v + mad_h)

        for sf in std_factors:
            for di in dilates:
                for ke in k_extras:
                    for tf in tol_factors:
                        res = _detect_grid_lines(
                            img_rgba,
                            min_pitch=min_pitch,
                            max_pitch=max_pitch,
                            std_factor=sf,
                            dilate_iters=di,
                            k_extra=ke,
                            tol_factor=tf,
                        )
                        sc = _score(res)
                        if sc > best_score:
                            best_score = sc
                            best = res

        if best is not None and best.get('ok'):
            return best

        # 如果最佳结果仍不稳定，使用更激进参数在窄窗内重试
        retry = _detect_grid_lines(
            img_rgba,
            min_pitch=min_pitch,
            max_pitch=max_pitch,
            std_factor=1.0,
            dilate_iters=2,
            k_extra=2,
            tol_factor=0.35,
        )
        if retry.get('ok'):
            return retry

        return best if best is not None else {
            'ok': False, 'v_lines': [], 'h_lines': [], 'pitch_w': 0, 'pitch_h': 0, 'preview': None
        }
    except Exception:
        return {'ok': False, 'v_lines': [], 'h_lines': [], 'pitch_w': 0, 'pitch_h': 0, 'preview': None}


def _build_uniform_grid(img_rgba: np.ndarray, pitch: int):
    """在已知统一格距情况下，依据投影最大响应估计相位偏移，并重建整张网格。

    返回与检测一致的结构：v_lines、h_lines、pitch_* 与预览。
    """
    try:
        h, w = img_rgba.shape[:2]
        if pitch is None or int(pitch) <= 0:
            return {'ok': False, 'v_lines': [], 'h_lines': [], 'pitch_w': 0, 'pitch_h': 0, 'preview': None}
        pitch = int(pitch)
        # 生成二值图以计算列/行投影（与检测保持一致）
        bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.medianBlur(gray, 3)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21, 2
        )
        col_sum = (binary // 255).sum(axis=0).astype(float)
        row_sum = (binary // 255).sum(axis=1).astype(float)

        def _best_offset(proj: np.ndarray, p: int):
            p = int(p)
            if p <= 0:
                return 0
            scores = [proj[o::p].sum() for o in range(min(p, len(proj)))]
            return int(np.argmax(scores)) if scores else 0

        off_x = _best_offset(col_sum, pitch)
        off_y = _best_offset(row_sum, pitch)
        v_lines = [int(x) for x in range(off_x, w, pitch)]
        h_lines = [int(y) for y in range(off_y, h, pitch)]
        ok = len(v_lines) >= 3 and len(h_lines) >= 3

        # 预览
        pil_preview = Image.fromarray(img_rgba.astype(np.uint8), 'RGBA').convert('RGB')
        draw = ImageDraw.Draw(pil_preview)
        for x in v_lines:
            draw.line([(x, 0), (x, h)], fill=(255, 0, 0), width=1)
        for y in h_lines:
            draw.line([(0, y), (w, y)], fill=(0, 128, 255), width=1)

        return {
            'ok': ok,
            'v_lines': v_lines,
            'h_lines': h_lines,
            'pitch_w': pitch,
            'pitch_h': pitch,
            'preview': pil_preview
        }
    except Exception:
        return {'ok': False, 'v_lines': [], 'h_lines': [], 'pitch_w': 0, 'pitch_h': 0, 'preview': None}


def _annotate_grid_with_codes(img_rgba: np.ndarray, v_lines: list[int], h_lines: list[int], palette_name: str):
    """根据网格线将图像划分为网格，计算每格代表色并匹配到调色板编码，生成标注图和统计。

    返回：{'image': PIL.Image, 'counts': Dict[str, int]}
    """
    try:
        h, w = img_rgba.shape[:2]
        # 构造标注底图并叠加网格线
        pil_annot = Image.fromarray(img_rgba.astype(np.uint8), 'RGBA').convert('RGB')
        draw = ImageDraw.Draw(pil_annot)
        for x in v_lines:
            draw.line([(x, 0), (x, h)], fill=(255, 0, 0), width=1)
        for y in h_lines:
            draw.line([(0, y), (w, y)], fill=(0, 128, 255), width=1)

        # 计算网格单元边界（排除线本身1px，尽量取有效内容）
        xs = [0] + [min(max(0, int(x)), w - 1) for x in v_lines] + [w]
        ys = [0] + [min(max(0, int(y)), h - 1) for y in h_lines] + [h]

        cu = ColorUtils()
        counts: dict[str, int] = {}
        approx_counts: dict[str, int] = {}

        arr = np.array(pil_annot)
        font = ImageFont.load_default()

        for j in range(len(ys) - 1):
            y0 = ys[j]
            y1 = ys[j + 1]
            # 去掉网格线像素
            y0_eff = min(h - 1, max(0, y0 + 1))
            y1_eff = max(0, min(h, y1 - 1))
            if y1_eff <= y0_eff:
                continue
            for i in range(len(xs) - 1):
                x0 = xs[i]
                x1 = xs[i + 1]
                x0_eff = min(w - 1, max(0, x0 + 1))
                x1_eff = max(0, min(w, x1 - 1))
                if x1_eff <= x0_eff:
                    continue

                region = arr[y0_eff:y1_eff, x0_eff:x1_eff]
                if region.size == 0:
                    continue
                # 使用非透明像素的平均色作为该格代表色
                # 注：当前为RGB图，默认无透明；若未来用RGBA，可筛选alpha>0
                rgb_mean = region[:, :, :3].reshape(-1, 3).mean(axis=0)
                target_rgb = (int(rgb_mean[0]), int(rgb_mean[1]), int(rgb_mean[2]))
                closest = cu.find_closest_palette_color(target_rgb, palette_name)
                code = closest.get('name', '')
                counts[code] = counts.get(code, 0) + 1
                # 近似标记：当距离>0则认为不是该品牌调色板的精确色，做近似计数
                dist = float(closest.get('distance', 1.0))
                if dist > 0.0:
                    approx_counts[code] = approx_counts.get(code, 0) + 1

                # 在网格中心写入编码（居中），文本颜色依据背景亮度自动选择
                cx = (x0_eff + x1_eff) // 2
                cy = (y0_eff + y1_eff) // 2
                brightness = (target_rgb[0] * 299 + target_rgb[1] * 587 + target_rgb[2] * 114) / 1000.0
                text_color = (255, 255, 255) if brightness < 140 else (0, 0, 0)
                bbox = draw.textbbox((0, 0), code, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                tx = cx - tw // 2
                ty = cy - th // 2
                draw.text((tx, ty), code, fill=text_color, font=font)

        return {'image': pil_annot, 'counts': counts, 'approx': approx_counts}
    except Exception:
        return {'image': None, 'counts': {}, 'approx': {}}

def main():
    st.title("📐 网格线识别与预览")
    st.markdown("上传拼豆图纸或网格图片，进行网格线自动识别并查看预览。")

    # 左侧导航栏：全局设置（识别模式、拼豆品牌）
    with st.sidebar:
        st.header("全局设置")
        # 识别模式选择：自动识别 或 统一格距识别
        mode_options = ["自动识别", "统一格距识别"]
        mode_idx = int(st.session_state.get('mode_idx', 0))
        mode_select = st.radio(
            "识别模式",
            options=mode_options,
            index=mode_idx if 0 <= mode_idx < len(mode_options) else 0,
            help="选择识别方式：自动识别网格线，或按统一格距重建整齐网格"
        )
        st.session_state.mode_idx = mode_options.index(mode_select)

        # 当选择统一格距识别时，提供格距输入（像素）
        if mode_select == "统一格距识别":
            uniform_pitch_default = int(st.session_state.get('uniform_pitch', 32))
            uniform_pitch = st.number_input(
                "统一格距（像素）",
                min_value=5,
                max_value=500,
                value=uniform_pitch_default,
                step=1,
                help="设置每格的像素间距（建议与图片实际网格一致）"
            )
            st.session_state.uniform_pitch = int(uniform_pitch)
            # 显示自动估计的建议值（若存在）
            if 'uniform_pitch_suggested' in st.session_state and int(st.session_state.uniform_pitch_suggested) > 0:
                st.caption(f"建议值：{int(st.session_state.uniform_pitch_suggested)} 像素（已自动填充，可手动调整）")

        # 拼豆品牌（跨页面全局使用，保留在左侧导航）
        try:
            palette_names = get_palette_names()
        except Exception:
            palette_names = ["perler", "artkal"]
        if 'selected_palette' not in st.session_state:
            st.session_state.selected_palette = palette_names[0]
        selected_palette = st.selectbox(
            "拼豆品牌",
            options=palette_names,
            index=palette_names.index(st.session_state.selected_palette) if st.session_state.selected_palette in palette_names else 0,
            key="global_palette_select",
            help="选择拼豆品牌（全局设置）"
        )
        st.session_state.selected_palette = selected_palette

    # 上传与参数设置
    mosaic_file = st.file_uploader("选择图片", type=['png', 'jpg', 'jpeg', 'bmp'], key="mosaic_uploader")

    # 已移除高级参数设置；通过侧边栏选择识别模式

    if mosaic_file is not None:
        # 识别不同上传文件，避免重复估计
        current_name = getattr(mosaic_file, 'name', None)
        if st.session_state.get('last_mosaic_name') != current_name:
            st.session_state.last_mosaic_name = current_name
            st.session_state.auto_pitch_initialized = False

        mosaic_img = Image.open(mosaic_file).convert('RGBA')
        st.image(mosaic_img, caption="拼豆图纸（原图）", use_container_width=True)

        # 上传后自动估计统一格距（仅一次），填充到侧边栏输入框
        img_array = np.array(mosaic_img)
        if not bool(st.session_state.get('auto_pitch_initialized', False)):
            try:
                auto_res = _auto_detect_grid_lines(img_array)
                pitch_suggested = max(auto_res.get('pitch_w', 0), auto_res.get('pitch_h', 0))
                if pitch_suggested and int(pitch_suggested) > 0:
                    st.session_state.uniform_pitch = int(pitch_suggested)
                    st.session_state.uniform_pitch_suggested = int(pitch_suggested)
                else:
                    st.session_state.uniform_pitch_suggested = 0
                st.session_state.auto_pitch_initialized = True
            except Exception:
                st.session_state.uniform_pitch_suggested = 0
                st.session_state.auto_pitch_initialized = True

        # 识别网格线并确认
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            # 根据模式展示不同的识别操作按钮
            if int(st.session_state.get('mode_idx', 0)) == 0:
                # 自动识别模式
                if st.button("自动识别网格线", key="auto_detect_grid_btn"):
                    result = _auto_detect_grid_lines(img_array)
                    st.session_state.grid_detect = result
            else:
                # 统一格距识别模式
                pitch_val = int(st.session_state.get('uniform_pitch', 0))
                if st.button("按统一格距识别并重建", key="uniform_grid_btn"):
                    result = _build_uniform_grid(img_array, pitch_val)
                    st.session_state.grid_detect = result
        with col_b2:
            if st.button("清除识别结果", key="clear_grid_btn"):
                st.session_state.grid_detect = {'ok': False, 'v_lines': [], 'h_lines': [], 'pitch_w': 0, 'pitch_h': 0, 'preview': None}

        if 'grid_detect' in st.session_state and st.session_state.grid_detect.get('preview') is not None:
            gd = st.session_state.grid_detect
            st.image(gd['preview'], caption=f"识别预览（红=竖线, 蓝=横线）｜pitch: {gd['pitch_w']}×{gd['pitch_h']}", use_container_width=True)
            if gd['ok']:
                st.success("检测到规则网格线。")
            else:
                st.warning("未能稳定检测到网格线（或间距不规则）。")

            # 网格编码与统计
            st.subheader("📋 网格颜色编码与数量统计")
            if st.button("填入编码并统计", key="annotate_and_stats_btn"):
                try:
                    palette = st.session_state.get('selected_palette', 'perler')
                    ann = _annotate_grid_with_codes(
                        np.array(mosaic_img),
                        gd.get('v_lines', []),
                        gd.get('h_lines', []),
                        palette,
                    )
                    if ann.get('image') is not None:
                        st.session_state.grid_annotated_img = ann['image']
                        st.session_state.grid_counts = ann['counts']
                except Exception:
                    st.session_state.grid_annotated_img = None
                    st.session_state.grid_counts = {}

            if st.session_state.get('grid_annotated_img') is not None:
                st.image(st.session_state.grid_annotated_img, caption="网格编码标注图", use_container_width=True)

            if st.session_state.get('grid_counts'):
                counts = st.session_state.grid_counts
                approx = st.session_state.get('grid_counts_approx', {})
                # 构建颜色统计表格（编码+色块）
                try:
                    palette = st.session_state.get('selected_palette', 'perler')
                    palette_colors = get_palette_colors(palette)
                except Exception:
                    palette_colors = {}
                total_cells = sum(counts.values())
                rows_html = []
                # 按数量降序
                for code, cnt in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
                    hex_color = palette_colors.get(code)
                    if hex_color:
                        hex_color = ColorUtils.normalize_hex(hex_color)
                    else:
                        hex_color = '#CCCCCC'
                    cell = (
                        f"<div style='display:flex;align-items:center;gap:8px'>"
                        f"<span style='display:inline-block;width:24px;height:24px;border:1px solid #ccc;background-color:{hex_color}'></span>"
                        f"<code>{code}</code>"
                        f"</div>"
                    )
                    percent = f"{(cnt/total_cells*100):.1f}%"
                    approx_cnt = int(approx.get(code, 0))
                    remark = (f"近似 {int(round(approx_cnt/cnt*100))}%" if cnt > 0 and approx_cnt > 0 else "")
                    rows_html.append(f"<tr><td>{cell}</td><td>{cnt}</td><td>{percent}</td><td>{remark}</td></tr>")
                table_html = (
                    "<table style='width:100%;border-collapse:collapse'>"
                    "<thead><tr>"
                    "<th style='text-align:left;padding:6px;border-bottom:1px solid #ddd'>颜色编码</th>"
                    "<th style='text-align:left;padding:6px;border-bottom:1px solid #ddd'>数量</th>"
                    "<th style='text-align:left;padding:6px;border-bottom:1px solid #ddd'>百分比</th>"
                    "<th style='text-align:left;padding:6px;border-bottom:1px solid #ddd'>备注</th>"
                    "</tr></thead>"
                    f"<tbody>{''.join(rows_html)}</tbody>"
                    "</table>"
                )
                st.markdown(table_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()