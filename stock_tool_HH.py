import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 强制绕过代理
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

st.set_page_config(page_title="DKX 极速选股", layout="wide")

# --- 核心算法 ---
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

# --- 核心扫描任务 ---
def scan_stock(code, name, n, m, threshold, select_mode, adj):
    try:
        # 手机端减少数据请求量，只拿最近100天，加快速度
        df_hist = ak.stock_zh_a_hist(symbol=code, period="daily", adjust=adj)
        if len(df_hist) < (n + m + 5): return None
        
        df_hist = calculate_dkx_final(df_hist, n, m)
        last = df_hist.iloc[-1]
        
        diff = abs(last['dkx'] - last['madkx'])
        raw_diff = last['dkx'] - last['madkx']
        
        if diff > threshold: return None
        if select_mode == "即将上穿" and raw_diff >= 0: return None
        if select_mode == "已经上穿" and raw_diff <= 0: return None
            
        return {
            "代码": code, "名称": name, "现价": last['close'],
            "DKX": round(last['dkx'], 2), "MADKX": round(last['madkx'], 2),
            "绝对差值": round(diff, 2), "日期": last['date']
        }
    except:
        return None

# --- UI ---
st.title("🏹 DKX 手机增强版")

with st.sidebar:
    st.header("1. 范围与市值")
    pool_type = st.selectbox("股票池", ["沪深300", "中证500", "全A股"])
    mv_min, mv_max = st.slider("市值范围(亿)", 0, 2000, (500, 1000))
    
    st.header("2. 技术参数")
    p_n = st.number_input("DKX(N)", value=20)
    p_m = st.number_input("MADKX(M)", value=10)
    limit = st.number_input("差值阈值", value=2.0)
    mode = st.selectbox("形态", ["全部满足", "即将上穿", "已经上穿"])
    adj = st.selectbox("复权", ["qfq", ""])

if st.button("🚀 开始扫描 (手机建议选300/500)", type="primary"):
    results = []
    
    # 第一步：获取名单与市值 (使用最稳的接口)
    with st.status("正在初始化数据源...", expanded=True) as status:
        st.write("📡 正在抓取全市场快照...")
        try:
            market_data = ak.stock_zh_a_spot_em()
            # 过滤市值
            market_data['total_mv_billion'] = market_data['总市值'] / 1e8
            filtered_df = market_data[(market_data['total_mv_billion'] >= mv_min) & (market_data['total_mv_billion'] <= mv_max)]
            
            # 过滤股票池
            if pool_type == "沪深300":
                st.write("🔍 提取沪深300成员...")
                cons_df = ak.index_stock_cons_weight_csindex(symbol="000300")
                target_codes = cons_df['成分券代码'].tolist()
                filtered_df = filtered_df[filtered_df['代码'].isin(target_codes)]
            elif pool_type == "中证500":
                st.write("🔍 提取中证500成员...")
                cons_df = ak.index_stock_cons_weight_csindex(symbol="000905")
                target_codes = cons_df['成分券代码'].tolist()
                filtered_df = filtered_df[filtered_df['代码'].isin(target_codes)]
            
            st.write(f"✅ 待分析目标: {len(filtered_df)} 只")
        except Exception as e:
            st.error(f"获取名单失败: {e}")
            st.stop()

        # 第二步：多线程分析技术面
        st.write("📊 开始计算技术指标...")
        progress_bar = st.progress(0)
        
        # 降低线程数到 5，防止被云端封禁 IP 或内存溢出
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_stock = {
                executor.submit(scan_stock, row['代码'], row['名称'], p_n, p_m, limit, mode, adj): row['代码'] 
                for _, row in filtered_df.iterrows()
            }
            
            done_count = 0
            for future in as_completed(future_to_stock):
                done_count += 1
                res = future.result()
                if res:
                    results.append(res)
                progress_bar.progress(done_count / len(filtered_df))
        
        status.update(label="扫描完成!", state="complete", expanded=False)

    # 展示结果
    if results:
        res_df = pd.DataFrame(results)
        st.success(f"找到 {len(results)} 只个股")
        st.dataframe(res_df, use_container_width=True)
    else:
        st.warning("未找到符合条件的股票。")
