import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# 屏蔽代理
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

st.set_page_config(page_title="DKX 极速选股 Pro", layout="wide")

# --- 1. 带缓存的数据抓取函数 ---
@st.cache_data(ttl=3600)
def get_pool_list(pool_type):
    try:
        if pool_type == "沪深300":
            df = ak.index_stock_cons_weight_csindex(symbol="000300")
            return dict(zip(df['成分券代码'], df['成分券名称']))
        elif pool_type == "中证500":
            df = ak.index_stock_cons_weight_csindex(symbol="000905")
            return dict(zip(df['成分券代码'], df['成分券名称']))
        elif pool_type == "全A股":
            df = ak.stock_zh_a_spot_em()
            return dict(zip(df['代码'], df['名称']))
    except:
        return {}
    return {}

@st.cache_data(ttl=1800)
def get_market_snapshot():
    try:
        df = ak.stock_zh_a_spot_em()
        df['total_mv_billion'] = df['总市值'] / 1e8
        return df[['代码', '名称', 'total_mv_billion']]
    except:
        return pd.DataFrame()

# --- 2. 核心算法 ---
def calculate_dkx_fast(df, n, m):
    cols = {'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low'}
    df = df.rename(columns=cols)
    df['mid'] = (3 * df['close'] + df['low'] + df['open'] + df['high']) / 6
    weights = np.arange(n, 0, -1)
    sum_w = np.sum(weights)
    def dkx_val(s):
        return np.dot(s, weights[::-1]) / sum_w if len(s) == n else np.nan
    df['dkx'] = df['mid'].rolling(window=n).apply(dkx_val, raw=True)
    df['madkx'] = df['dkx'].rolling(window=m).mean()
    return df

def scan_worker(code, name, n, m, limit, mode, adj, start_date):
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, adjust=adj)
        if len(df) < (n + m + 2): return None
        df = calculate_dkx_fast(df, n, m)
        last = df.iloc[-1]
        diff = abs(last['dkx'] - last['madkx'])
        raw_diff = last['dkx'] - last['madkx']
        if diff > limit: return None
        if mode == "即将上穿" and raw_diff >= 0: return None
        if mode == "已经上穿" and raw_diff <= 0: return None
        return {
            "代码": code, "名称": name, "现价": last['close'],
            "DKX": round(last['dkx'], 2), "MADKX": round(last['madkx'], 2),
            "差值": round(diff, 2), "更新日期": last['date']
        }
    except:
        return None

# --- 3. UI 界面 ---
st.title("🏹 DKX 极速选股 Pro")

with st.sidebar:
    st.header("🔍 筛选配置")
    pool = st.selectbox("股票池", ["沪深300", "中证500", "全A股"], index=0)
    mv_range = st.slider("市值范围(亿)", 0, 5000, (100, 2000))
    
    st.divider()
    n_val = st.number_input("DKX(N)", 20)
    m_val = st.number_input("MADKX(M)", 10)
    # 【修复重点】设定最小值 min_value=0.01，确保你可以输入 0.1 等极小值
    limit_val = st.number_input("差值阈值 (越小越接近金叉)", min_value=0.01, value=0.5, step=0.1)
    mode_val = st.selectbox("形态", ["全部满足", "即将上穿", "已经上穿"])
    adj_val = st.selectbox("复权", ["qfq", ""])

if st.button("🚀 开始极速扫描", type="primary"):
    results = []
    start_dt = (datetime.now() - timedelta(days=120)).strftime("%Y%m%d")
    with st.status("正在分析数据...", expanded=True) as status:
        st.write("📂 加载数据快照...")
        snapshot = get_market_snapshot()
        pool_dict = get_pool_list(pool)
        if snapshot.empty or not pool_dict:
            st.error("数据抓取失败")
            st.stop()
        filtered_codes = [
            (c, n) for c, n in pool_dict.items() 
            if c in snapshot['代码'].values and 
            mv_range[0] <= snapshot.loc[snapshot['代码']==c, 'total_mv_billion'].values[0] <= mv_range[1]
        ]
        st.write(f"🧪 扫描目标: {len(filtered_codes)} 只")
        progress = st.progress(0)
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(scan_worker, c, n, n_val, m_val, limit_val, mode_val, adj_val, start_dt) for c, n in filtered_codes]
            for i, f in enumerate(as_completed(futures)):
                res = f.result()
                if res: results.append(res)
                progress.progress((i + 1) / len(filtered_codes))
        status.update(label=f"扫描完毕！找到 {len(results)} 个目标", state="complete")

    if results:
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.warning("未找到匹配项。")
