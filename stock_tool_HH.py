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

st.set_page_config(page_title="DKX 极速多因子选股器", layout="wide")

# --- 1. 数据抓取与缓存 ---
@st.cache_data(ttl=3600)
def get_stock_pool_data(pool_type):
    """获取名单和基本面混合数据"""
    try:
        # 获取实时行情（含市值）
        df_spot = ak.stock_zh_a_spot_em()
        df_spot['total_mv_billion'] = df_spot['总市值'] / 1e8
        
        # 获取最新季度业绩报表 (营收增长、净利润)
        # 自动尝试最新年份，若报错则退回上一个季度
        try:
            df_finance = ak.stock_yjbb_em(date="20241231")
        except:
            df_finance = ak.stock_yjbb_em(date="20240930")
            
        df_finance = df_finance[['股票代码', '营业收入-同比增长', '净利润-净利润']]
        df_finance.columns = ['代码', '营收同比', '净利润(亿)']
        df_finance['净利润(亿)'] = df_finance['净利润(亿)'] / 1e8
        
        # 合并行情与财务
        df_combined = pd.merge(df_spot, df_finance, on='代码', how='left')
        
        # 过滤股票池成员
        if pool_type == "沪深300":
            cons = ak.index_stock_cons_weight_csindex(symbol="000300")
            df_combined = df_combined[df_combined['代码'].isin(cons['成分券代码'].tolist())]
        elif pool_type == "中证500":
            cons = ak.index_stock_cons_weight_csindex(symbol="000905")
            df_combined = df_combined[df_combined['代码'].isin(cons['成分券代码'].tolist())]
            
        return df_combined
    except:
        return pd.DataFrame()

# --- 2. 算法引擎 ---
def calculate_dkx_logic(df, n, m):
    df = df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low'})
    df['mid'] = (3 * df['close'] + df['low'] + df['open'] + df['high']) / 6
    weights = np.arange(n, 0, -1)
    sum_w = np.sum(weights)
    def dkx_val(s): return np.dot(s, weights[::-1]) / sum_w if len(s) == n else np.nan
    df['dkx'] = df['mid'].rolling(window=n).apply(dkx_val, raw=True)
    df['madkx'] = df['dkx'].rolling(window=m).mean()
    return df

def scan_worker(row, n, m, limit, mode, adj, start_date):
    try:
        code = row['代码']
        df_hist = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, adjust=adj)
        if len(df_hist) < (n + m + 2): return None
        df_hist = calculate_dkx_logic(df_hist, n, m)
        last = df_hist.iloc[-1]
        diff = abs(last['dkx'] - last['madkx'])
        raw_diff = last['dkx'] - last['madkx']
        
        if diff > limit: return None
        if mode == "即将上穿" and raw_diff >= 0: return None
        if mode == "已经上穿" and raw_diff <= 0: return None
        
        return {
            "代码": code, "名称": row['名称'], "现价": last['close'],
            "DKX": round(last['dkx'], 3), "MADKX": round(last['madkx'], 3),
            "绝对差值": round(diff, 4), "营收同比%": row['营收同比'],
            "净利润(亿)": round(row['净利润(亿)'], 2), "日期": last['date']
        }
    except: return None

# --- 3. UI 界面 ---
st.title("🏹 DKX & 财务因子综合选股")

with st.sidebar:
    st.header("🎯 指标精度设置")
    limit_val = st.number_input("DKX差值阈值 (精度0.01)", min_value=0.001, value=0.050, step=0.010, format="%.3f")
    mode_val = st.selectbox("DKX形态", ["全部满足", "即将上穿", "已经上穿"])
    
    st.header("📊 财务因子筛选")
    min_rev = st.number_input("营收同比增长 > (%)", value=10.0)
    min_profit = st.number_input("净利润额 > (亿)", value=1.0)
    
    st.header("🏢 市场与范围")
    pool = st.selectbox("股票池", ["沪深300", "中证500", "全A股"])
    mv_range = st.slider("总市值范围(亿)", 0, 5000, (100, 2000))
    adj_val = st.selectbox("复权方式", ["qfq", ""])

if st.button("🚀 开始极速多因子扫描", type="primary"):
    start_dt = (datetime.now() - timedelta(days=120)).strftime("%Y%m%d")
    
    with st.status("多维度分析中...", expanded=True) as status:
        st.write("📡 正在获取行情及财报快照...")
        data_all = get_stock_pool_data(pool)
        
        if data_all.empty:
            st.error("数据抓取异常，请重试")
            st.stop()
            
        # 执行财务与市值预筛选
        st.write("🧪 正在执行财务因子过滤...")
        pre_filtered = data_all[
            (data_all['total_mv_billion'].between(mv_range[0], mv_range[1])) &
            ((data_all['营收同比'] >= min_rev) | (data_all['营收同比'].isna())) & # 读不到财报的默认通过或根据原则跳过
            ((data_all['净利润(亿)'] >= min_profit) | (data_all['净利润(亿)'].isna()))
        ]
        
        st.write(f"✅ 进入技术面复核: {len(pre_filtered)} 只")
        
        results = []
        progress = st.progress(0)
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(scan_worker, row, 20, 10, limit_val, mode_val, adj_val, start_dt) for _, row in pre_filtered.iterrows()]
            for i, f in enumerate(as_completed(futures)):
                res = f.result()
                if res: results.append(res)
                progress.progress((i + 1) / len(pre_filtered))
        
        status.update(label=f"分析完毕! 找到 {len(results)} 只个股", state="complete")

    if results:
        st.balloons()
        df_res = pd.DataFrame(results).sort_values(by="绝对差值")
        st.dataframe(df_res, use_container_width=True)
    else:
        st.warning("在此严苛条件下未找到匹配个股，建议降低财务要求或增大差值阈值。")
