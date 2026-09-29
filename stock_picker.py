import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
import datetime
import os
import time

# ================= 页面配置 (手机端深度优化) =================
st.set_page_config(layout="wide", page_title="A股多周期选股器", page_icon="📈")

st.markdown("""
<style>
    @media (max-width: 768px) {
        .block-container { padding: 1rem 0.5rem !important; }
        h1 { font-size: 1.4rem !important; }
        .stTabs [data-baseweb="tab"] { font-size: 0.85rem !important; padding: 0.5rem !important; }
        .stCheckbox { font-size: 0.9rem !important; }
        .stSelectSlider { font-size: 0.9rem !important; }
    }
</style>
""", unsafe_allow_html=True)

DATA_DIR = "stock_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# ================= 核心指标计算 (DKX算法已修正) =================

def compute_dkx_macd_ma(df):
    """接收包含 date, open, close, high, low, volume 的DataFrame，计算所有指标"""
    # 1. DKX & MADKX (修正后的标准算法，与新浪财经一致)
    # 标准公式：DKX = (20*MID_t + 19*MID_{t-1} + ... + 1*MID_{t-19}) / 210
    df['mid'] = (3 * df['close'] + df['low'] + df['open'] + df['high']) / 6
    weights = np.arange(20, 0, -1)  # [20, 19, ..., 1]
    sum_w = np.sum(weights)         # 210
    
    def dkx_val(s):
        if len(s) < 20: return np.nan
        # s 是时间正序(最老->最新)，s[::-1] 得到 [1,2,...,20] 权重，即最老乘1，最新乘20
        return np.dot(s, weights[::-1]) / sum_w
        
    df['DKX'] = df['mid'].rolling(window=20).apply(dkx_val, raw=True)
    df['MADKX'] = df['DKX'].rolling(window=10).mean() # M=10

    # 2. MACD
    ema_short = df['close'].ewm(span=12, adjust=False).mean()
    ema_long = df['close'].ewm(span=26, adjust=False).mean()
    dif = ema_short - ema_long
    dea = dif.ewm(span=9, adjust=False).mean()
    df['MACD'] = (dif - dea) * 2

    # 3. 均线
    for period in [5, 10, 20, 30, 60]:
        df[f'MA{period}'] = df['close'].rolling(window=period).mean()

    # 4. 量比
    vol_ma5 = df['volume'].shift(1).rolling(5).mean()
    df['量比'] = df['volume'] / vol_ma5

    return df

def resample_data(df, period='W'):
    """重采样生成周线或月线。df 需为中文列名"""
    df_reset = df.reset_index()
    df_reset = df_reset.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'})
    df_reset = df_reset.set_index('date')
    
    resampled = df_reset.resample(period).agg({
        'open': 'first', 'close': 'last', 'high': 'max', 'low': 'min', 'volume': 'sum'
    }).dropna()
    
    resampled = compute_dkx_macd_ma(resampled)
    resampled = resampled.rename(columns={'open': '开盘', 'close': '收盘', 'high': '最高', 'low': '最低', 'volume': '成交量'})
    return resampled

# ================= 数据获取与存储 =================

@st.cache_data(ttl=3600*24)
def get_stock_list(token):
    pro = ts.pro_api(token)
    try:
        df = pro.stock_basic(exchange='', list_status='L', fields='symbol,name')
        return df.rename(columns={'symbol': '代码', 'name': '名称'})
    except Exception as e:
        st.error(f"获取股票列表失败: {e}")
        return pd.DataFrame()

def fetch_and_update_stock(code, token, start_date_str="20230101"):
    """全量下载单只股票的历史数据，覆盖写入Excel"""
    ts_code = code + '.SZ' if code.startswith(('0', '3')) else code + '.SH'
    file_path = os.path.join(DATA_DIR, f"{code}.xlsx")
    end_date = datetime.datetime.now().strftime('%Y%m%d')
    
    ts.set_token(token)
    try:
        df_new = ts.pro_bar(ts_code=ts_code, start_date=start_date_str, end_date=end_date, adj='qfq')
        if df_new is None or len(df_new) < 60: # 数据少于60天跳过
            return 0
            
        df_new = df_new.rename(columns={'trade_date': '日期', 'open': '开盘', 'close': '收盘', 'high': '最高', 'low': '最低', 'vol': '成交量'})
        df_new['日期'] = pd.to_datetime(df_new['日期'])
        df_new = df_new.sort_values('日期')
        df_new = df_new[['日期', '开盘', '收盘', '最高', '最低', '成交量']]
        
        df_combined = df_new.set_index('日期')
        df_calc = df_combined.reset_index().rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'})
        df_calc = compute_dkx_macd_ma(df_calc)
        df_final = df_calc.rename(columns={'date': '日期', 'open': '开盘', 'close': '收盘', 'high': '最高', 'low': '最低', 'volume': '成交量'})
        
        df_weekly = resample_data(df_final, 'W-FRI')
        df_monthly = resample_data(df_final, 'ME')

        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df_final.to_excel(writer, sheet_name='日线', index=False)
            df_weekly.to_excel(writer, sheet_name='周线', index=False)
            df_monthly.to_excel(writer, sheet_name='月线', index=False)
        return len(df_new)
    except Exception as e:
        return -1

# ================= 选股逻辑 =================

def check_dkx_cross(df, mode="已经上穿", limit=0.05):
    if len(df) < 2: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    if pd.isna(last['DKX']) or pd.isna(last['MADKX']) or pd.isna(prev['DKX']) or pd.isna(prev['MADKX']):
        return False
    raw_diff = last['DKX'] - last['MADKX']
    if mode == "即将上穿":
        return raw_diff < 0 and abs(raw_diff) <= limit
    elif mode == "已经上穿":
        return prev['DKX'] <= prev['MADKX'] and raw_diff > 0
    return False

def check_macd_increasing(df, periods=3):
    if len(df) < periods + 1: return False
    recent = df['MACD'].dropna().tail(periods + 1)
    if len(recent) < periods + 1: return False
    return all(recent.iloc[i] > recent.iloc[i-1] for i in range(1, len(recent)))

def check_volume_ratio_increase(df, days=3, threshold=1.0):
    if len(df) < 6: return False
    return all(df['量比'].tail(days) > threshold)

# ================= 主界面 UI =================

st.title("📈 A股多周期选股器 (回测版)")
st.caption("Tushare Pro 数据源 | 支持导出Excel与增量更新")

with st.sidebar:
    st.header("🔑 数据源设置")
    token_input = st.text_input("Tushare Token", type="password")
    
    st.divider()
    st.header("📅 数据更新设置")
    start_year = st.selectbox("历史数据起始年份", ["2024", "2023", "2022", "2021", "2020"], index=1)
    start_date_str = f"{start_year}0101"
    update_btn = st.button("📥 全量下载/覆盖历史数据", type="primary", use_container_width=True)
    
    st.divider()
    st.header("🔎 初步筛选")
    exclude_st = st.checkbox("排除ST股票", value=True)
    mv_range = st.select_slider(
        "市值范围（亿元）",
        options=[0, 10, 20, 50, 100, 200, 500, 1000, 5000, 10000],
        value=(0, 5000)
    )
    min_mv, max_mv = mv_range[0] * 10000, mv_range[1] * 10000

    st.divider()
    st.header("⚙️ 选股条件")
    tab1, tab2, tab3 = st.tabs(["📊 多空线", "📈 MACD", "⚖️ 量价"])
    with tab1:
        cond1 = st.checkbox("日线DKX金叉", value=True)
        mode1 = st.radio("日线DKX模式", ["已经上穿", "即将上穿"], horizontal=True)
        cond2 = st.checkbox("周线DKX金叉", value=False)
    with tab2:
        cond3 = st.checkbox("日线MACD柱递增 (3天)", value=True)
        cond4 = st.checkbox("周线MACD柱递增 (3根)", value=False)
        cond5 = st.checkbox("月线MACD柱递增 (3根)", value=False)
    with tab3:
        cond6 = st.checkbox("量比连续大于1 (3天)", value=True)
    
    st.divider()
    execute_btn = st.button("🚀 执行选股", type="primary", use_container_width=True)
    reset_btn = st.button("重置", use_container_width=True)

# 数据更新逻辑
if update_btn:
    if not token_input: st.warning("请先输入Tushare Token！")
    else:
        stock_list = get_stock_list(token_input)
        if stock_list.empty: st.stop()
        total = len(stock_list)
        st.info(f"开始全量下载 {total} 只股票数据，存入Excel（请保持电脑常亮）...")
        progress = st.progress(0)
        status = st.empty()
        success_count = 0
        permission_warning = False
        
        for idx, row in stock_list.iterrows():
            progress.progress((idx+1)/total)
            status.text(f"下载: {row['名称']} ({row['代码']}) - {idx+1}/{total}")
            rows_count = fetch_and_update_stock(row['代码'], token_input, start_date_str)
            if rows_count > 0: success_count += 1
            elif rows_count == 0: permission_warning = True
            time.sleep(0.3)
            
        status.text("数据下载完成！")
        progress.empty()
        if permission_warning:
            st.error("⚠️ 警告：部分股票下载数据少于1年。这说明你的 Tushare 积分可能不足以拉取长期的前复权数据。请前往 Tushare 官网检查你的积分（建议升级到5000积分）。")
        else:
            st.success(f"成功下载 {success_count} 只股票的完整历史数据！")


# 选股逻辑
if execute_btn:
    if not token_input: st.warning("请先输入Tushare Token！")
    else:
        stock_list = get_stock_list(token_input)
        if stock_list.empty: st.stop()
        
        # ================= 关键修复：精确获取最新交易日 =================
        with st.spinner("正在获取最新交易日及初步筛选数据..."):
            pro = ts.pro_api(token_input)
            
            # 1. 获取最近的交易日（防止周末或节假日获取不到数据）
            today_str = datetime.datetime.now().strftime('%Y%m%d')
            start_search = (datetime.datetime.now() - datetime.timedelta(days=10)).strftime('%Y%m%d')
            cal_df = pro.trade_cal(exchange='SSE', is_open='1', start_date=start_search, end_date=today_str)
            latest_trade_date = cal_df['cal_date'].max() if cal_df is not None and not cal_df.empty else today_str
            
            st.info(f"使用最近交易日: {latest_trade_date} 获取市值与ST数据...")

            # 2. 获取市值
            mv_df = pro.daily_basic(trade_date=latest_trade_date, fields='ts_code,total_mv')
            if mv_df is not None and not mv_df.empty:
                mv_df['代码'] = mv_df['ts_code'].str[:6]
                mv_dict = dict(zip(mv_df['代码'], mv_df['total_mv']))
            else:
                mv_dict = {}
                st.error(f"⚠️ 无法获取 {latest_trade_date} 的市值数据，请检查 Tushare 积分权限（daily_basic需2000积分）。")
            
            # 3. 获取ST名单
            if exclude_st:
                st_df = pro.stock_basic(exchange='', list_status='L', fields='symbol,name')
                st_codes = set(st_df[st_df['name'].str.contains('ST')]['symbol'].tolist())
            else:
                st_codes = set()

            # 4. 执行筛选
            original_count = len(stock_list)
            
            if exclude_st and st_codes:
                stock_list = stock_list[~stock_list['代码'].isin(st_codes)]
            
            if mv_dict:
                stock_list['市值(万元)'] = stock_list['代码'].map(mv_dict)
                stock_list = stock_list.dropna(subset=['市值(万元)'])
                stock_list = stock_list[(stock_list['市值(万元)'] >= min_mv) & (stock_list['市值(万元)'] <= max_mv)]
            else:
                st.warning("未成功获取市值数据，已跳过市值筛选！")

        # 此时总数已经是过滤后的数量了
        total = len(stock_list)
        st.success(f"初步筛选完成：从 {original_count} 只股票中筛选出 {total} 只符合市值/ST条件的股票。")
        
        if total == 0:
            st.warning("没有股票符合初步筛选条件，请调整市值范围或取消排除ST选项。")
            st.stop()
            
        progress = st.progress(0)
        status = st.empty()
        results = []
        
        for idx, row in stock_list.iterrows():
            code, name = row['代码'], row['名称']
            progress.progress((idx+1)/total)
            status.text(f"筛选: {name} ({code}) - {idx+1}/{total}")
            
            # 下面保留你原有的选股逻辑
            file_path = os.path.join(DATA_DIR, f"{code}.xlsx")
            if not os.path.exists(file_path): continue
                
            try:
                df_daily = pd.read_excel(file_path, sheet_name='日线')
                df_weekly = pd.read_excel(file_path, sheet_name='周线')
                df_monthly = pd.read_excel(file_path, sheet_name='月线')
                
                if len(df_daily) < 60 or len(df_weekly) < 30 or len(df_monthly) < 10: continue
                    
                match = True
                reasons = []
                
                if cond1:
                    if check_dkx_cross(df_daily, mode1): reasons.append(f"日线金叉({mode1})")
                    else: match = False
                if match and cond2:
                    if check_dkx_cross(df_weekly, "已经上穿"): reasons.append("周线金叉")
                    else: match = False
                if match and cond3:
                    if check_macd_increasing(df_daily, 3): reasons.append("日线MACD递增")
                    else: match = False
                if match and cond4:
                    if check_macd_increasing(df_weekly, 3): reasons.append("周线MACD递增")
                    else: match = False
                if match and cond5:
                    if check_macd_increasing(df_monthly, 3): reasons.append("月线MACD递增")
                    else: match = False
                if match and cond6:
                    if check_volume_ratio_increase(df_daily, 3, 1.0): reasons.append("量比>1")
                    else: match = False
                
                if match:
                    last, prev = df_daily.iloc[-1], df_daily.iloc[-2]
                    change_pct = ((last['收盘'] - prev['收盘']) / prev['收盘']) * 100
                    results.append({
                        '代码': code, '名称': name,
                        '最新价': round(last['收盘'], 2),
                        '涨跌幅': f"{change_pct:+.2f}%",
                        '匹配条件': " · ".join(reasons)
                    })
            except Exception as e:
                continue
                
        status.text("筛选完成！")
        progress.empty()
        
        st.subheader(f"🎯 选出 {len(results)} 只股票")
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
        else:
            st.info("没有找到符合条件的股票，请尝试放宽条件。")

st.divider()
st.caption("提示：本工具数据来源于Tushare Pro，回测数据保存在 stock_data 文件夹。")
