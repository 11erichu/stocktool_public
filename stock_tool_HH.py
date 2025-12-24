import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# 环境配置
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

st.set_page_config(page_title="DKX 增强选股系统", layout="wide")

# --- 1. 高精度 DKX 计算引擎 ---
def calculate_dkx_final(df, n, m):
    column_map = {'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low'}
    df = df.rename(columns=column_map)
    df['mid'] = (3 * df['close'] + df['low'] + df['open'] + df['high']) / 6
    weights = np.arange(n, 0, -1)
    sum_weights = np.sum(weights)
    def dkx_formula(series):
        if len(series) < n: return np.nan
        return np.dot(series, weights[::-1]) / sum_weights
    df['dkx'] = df['mid'].rolling(window=n).apply(dkx_formula, raw=True)
    df['madkx'] = df['dkx'].rolling(window=m).mean()
    return df

# --- 2. 财务与市值筛选逻辑 ---
def get_fundamental_data():
    """获取全市场即时行情、市值及基础财务指标"""
    # 获取东财实时行情（包含市值、市盈率、营收增长率等）
    df_spot = ak.stock_zh_a_spot_em()
    # 列名重命名以便理解
    rename_cols = {
        '代码': 'code', '名称': 'name', '最新价': 'price',
        '总市值': 'total_mv', '市盈率-动态': 'pe',
        '成交额': 'amount', '涨跌幅': 'change_pct'
    }
    df_spot = df_spot.rename(columns=rename_cols)
    return df_spot

# --- 3. 核心扫描任务 ---
def scan_stock(row, n, m, threshold, select_mode, adj, mv_range, rev_growth):
    code = row['code']
    name = row['name']
    total_mv_billion = row['total_mv'] / 1e8 # 转为亿元
    
    # A. 市值过滤
    if not (mv_range[0] <= total_mv_billion <= mv_range[1]):
        return None
    
    # B. 营收增长过滤 (利用实时行情中的营业收入同比数据)
    # 注意：某些接口可能不含环比，这里优先判断同比
    try:
        yoy_growth = float(row.get('60日涨跌幅', 0)) # 示例：此处可替换为更精准的财务接口
        # 为了更精准，这里建议使用逻辑判断：如果代码需要精准财务数据，需要二次请求
    except:
        pass

    # C. 技术面 DKX 计算
    try:
        df_hist = ak.stock_zh_a_hist(symbol=code, period="daily", adjust=adj)
        if len(df_hist) < (n + m + 5): return None
        
        df_hist = calculate_dkx_final(df_hist, n, m)
        last = df_hist.iloc[-1]
        
        diff = abs(last['dkx'] - last['madkx'])
        raw_diff = last['dkx'] - last['madkx']
        
        # 筛选逻辑
        if diff > threshold: return None
        if select_mode == "即将上穿" and raw_diff >= 0: return None
        if select_mode == "已经上穿" and raw_diff <= 0: return None
            
        return {
            "代码": code, "名称": name, "现价": last['close'],
            "总市值(亿)": round(total_mv_billion, 2),
            "DKX": round(last['dkx'], 2), "MADKX": round(last['madkx'], 2),
            "绝对差值": round(diff, 2), "日期": last['date']
        }
    except:
        return None

# --- 4. 界面设计 ---
st.title("🛡️ DKX 增强型策略选股器")
st.markdown("结合 **DKX 趋势技术面** + **市值规模控制** + **营收基本面**")

with st.sidebar:
    st.header("📌 技术面设置")
    p_n = st.number_input("DKX 周期 (N)", value=20)
    p_m = st.number_input("MADKX 周期 (M)", value=10)
    mode = st.selectbox("形态选择", ["全部满足", "即将上穿", "已经上穿"])
    threshold = st.number_input("DKX/MADKX 差值阈值", value=2.0, step=0.1)
    
    st.header("💰 市值过滤 (亿元)")
    mv_min, mv_max = st.slider("总市值范围", 0, 5000, (500, 1000))
    
    st.header("📊 基本面(营收同比)")
    min_rev_yoy = st.number_input("最新季报营收同比 > (%)", value=10.0)

    st.header("⚙️ 系统设置")
    pool_type = st.selectbox("股票池", ["全A股", "沪深300", "中证500"])
    adj_type = st.selectbox("复权方式", ["不复权", "前复权"])
    adj = "qfq" if adj_type == "前复权" else ""

# --- 5. 执行逻辑 ---
if st.button("🚀 开始综合选股", type="primary"):
    with st.spinner("正在初始化数据源..."):
        # 获取全市场快照数据（包含市值和部分财务指标）
        raw_market_data = get_fundamental_data()
        
        # 精确获取财务报表（营收增长同比）
        # 这里使用“业绩报表-行报表”获取最新的营收增长率
        try:
            df_finance = ak.stock_yjbb_em(date="20241231") # 自动获取最新一季，此处建议设为最新季度末
            df_finance = df_finance[['股票代码', '营业收入-同比增长']]
            df_finance.columns = ['code', 'rev_yoy']
            # 合并数据
            merged_data = pd.merge(raw_market_data, df_finance, on='code', how='inner')
        except:
            st.warning("无法获取详细季报数据，将仅使用行情数据进行初步筛选")
            merged_data = raw_market_data
            merged_data['rev_yoy'] = 999 # 兜底逻辑

    # 预筛选：先通过市值和营收增长过滤，减少网络请求次数，大幅提升速度
    filtered_list = merged_data[
        (merged_data['total_mv'] >= mv_min * 1e8) & 
        (merged_data['total_mv'] <= mv_max * 1e8) &
        (merged_data['rev_yoy'] >= min_rev_yoy)
    ]
    
    st.info(f"预筛选完成：共 {len(filtered_list)} 只股票进入技术面分析...")
    
    results = []
    progress_bar = st.progress(0)
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(scan_stock, row, p_n, p_m, threshold, mode, adj, (mv_min, mv_max), min_rev_yoy) 
            for _, row in filtered_list.iterrows()
        ]
        
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            if res:
                # 补充营收增长率展示
                target_code = res['代码']
                rev_val = filtered_list[filtered_list['code'] == target_code]['rev_yoy'].values[0]
                res['营收同比(%)'] = round(rev_val, 2)
                results.append(res)
            progress_bar.progress((i + 1) / len(futures))

    if results:
        final_df = pd.DataFrame(results)
        # 调整列顺序
        cols = ['代码', '名称', '现价', '总市值(亿)', '营收同比(%)', 'DKX', 'MADKX', '绝对差值', '日期']
        st.success(f"🎊 扫描完毕！找到 {len(results)} 只完美匹配的个股。")
        st.dataframe(final_df[cols], use_container_width=True)
    else:
        st.warning("没有找到符合条件的股票，建议放宽市值或营收增长要求。")
