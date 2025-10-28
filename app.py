"""
Perler Bead Pattern Tool - Streamlit Web Interface
拼豆图案工具 - 欢迎页（多页面入口）
"""

import streamlit as st

# 页面配置集中于主入口，子页面不重复设置
st.set_page_config(
    page_title="拼豆图案工具",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎨 拼豆图案工具")
st.markdown("欢迎！请选择左侧页面开始使用。")

st.markdown(
    """
    - 生成拼豆图案：上传图片并转换为拼豆马赛克
    - 上传拼豆图纸统计：从图纸统计每种颜色的格子数量
    """
)

# 快速链接（Streamlit >=1.25 支持）
try:
    st.page_link("pages/generator.py", label="➡️ 前往：生成拼豆图案", icon="🎨")
    st.page_link("pages/mosaic_stats.py", label="➡️ 前往：上传拼豆图纸统计", icon="🧮")
except Exception:
    st.info("如果未显示页面链接，请使用左侧页面导航进入对应页面。")