import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
import datetime
import os
import time

# ================= 页面配置 (手机端适配) =================
st.set_page_config(layout="wide", page_title="A股多周期选股器", page_icon="📈")

st.markdown("""
<style>
    @media (max-width: 768px) {
        .block-container { padding: 1rem 0.5rem !important; }
        h1 { font-size: 1.4rem !important; }
        .stTabs [data-baseweb="tab"] { font-size: 0.85rem !important; padding: 0.5rem !important; }
        .stCheckbox { font-size: 0.9rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# 数据存放目录
DATA_DIR = "stock_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# ================= 核心指标计算 =================

def calculate_dkx_logic(df, n=10, m=10):
    """根据用户提供的算法计算DKX和MADKX"""
    # 重命名为英文进行计算
    df = df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low'})
    df['mid'] = (3 * df['close'] + df['low'] + df['open'] + df['high']) / 6
    weights = np.arange(n, 0, -1)
    sum_w = np.sum(weights)
    def dkx_val(s): return np.dot(s, weights[::-1]) / sum_w if len(s) == n else np.nan
    df['DKX'] = df['mid'].rolling(window=n).apply(dkx_val, raw=True)
    df['MADKX'] = df['DKX'].rolling(window=m).mean()
    
    # ===== 关键修复：把列名改回中文，供后续计算使用 =====
    df = df.rename(columns={'date': '日期', 'open': '开盘', 'close': '收盘', 'high': '最高', 'low': '最低'})
    return df

def calculate_macd(df, short=12, long=26, mid=9):
    """计算MACD"""
    ema_short = df['收盘'].ewm(span=short, adjust=False).mean()
    ema_long = df['收盘'].ewm(span=long, adjust=False).mean()
    dif = ema_short - ema_long
    dea = dif.ewm(span=mid, adjust=False).mean()
    df['MACD'] = (dif - dea) * 2
    return df

def calculate_ma(df):
    """计算均线"""
    for period in [5, 10, 20, 30, 60]:
        df[f'MA{period}'] = df['收盘'].rolling(window=period).mean()
    return df

def calculate_volume_ratio(df):
    """计算量比"""
    vol_ma5 = df['成交量'].shift(1).rolling(5).mean()
    df['量比'] = df['成交量'] / vol_ma5
    return df

def resample_data(df, period='W'):
    """重采样生成周线或月线"""
    resampled = df.resample(period).agg({
        '开盘': 'first', '收盘': 'last', '最高': 'max', '最低': 'min', '成交量': 'sum'
    }).dropna()
    # 为周线月线重新计算DKX和MACD
    resampled = calculate_dkx_logic(resampled)
    resampled = calculate_macd(resampled)
    return resampled

# ================= 数据获取与存储 =================

def get_stock_list(token):
    """获取全A股列表"""
    pro = ts.pro_api(token)
    try:
        df = pro.stock_basic(exchange='', list_status='L', fields='symbol,name')
        df = df.rename(columns={'symbol': '代码', 'name': '名称'})
        return df
    except Exception as e:
        st.error(f"获取股票列表失败: {e}")
        return pd.DataFrame()

def fetch_and_update_stock(code, token):
    """增量更新单只股票的数据，并保存到Excel"""
    ts_code = code + '.SZ' if code.startswith(('0', '3')) else code + '.SH'
    file_path = os.path.join(DATA_DIR, f"{code}.xlsx")
    
    end_date = datetime.datetime.now().strftime('%Y%m%d')
    
    # 判断是否需要增量更新
    if os.path.exists(file_path):
        try:
            df_old = pd.read_excel(file_path, sheet_name='日线')
            if df_old.empty:
                start_date = (datetime.datetime.now() - datetime.timedelta(days=3*365)).strftime('%Y%m%d')
            else:
                df_old['日期'] = pd.to_datetime(df_old['日期'])
                last_date = df_old['日期'].max()
                # 为了防止前复权数据错乱，每次更新拉取最近一年数据覆盖
                start_date = (last_date - datetime.timedelta(days=365)).strftime('%Y%m%d')
        except Exception:
            start_date = (datetime.datetime.now() - datetime.timedelta(days=3*365)).strftime('%Y%m%d')
    else:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=3*365)).strftime('%Y%m%d')

    # 调用Tushare获取数据 (pro_bar 需要全局token)
    ts.set_token(token)
    df_new = ts.pro_bar(ts_code=ts_code, start_date=start_date, end_date=end_date, adj='qfq')
    
    if df_new is None or df_new.empty:
        return None

    # 数据清洗与合并
    df_new = df_new.rename(columns={'trade_date': '日期', 'open': '开盘', 'close': '收盘', 'high': '最高', 'low': '最低', 'vol': '成交量'})
    df_new['日期'] = pd.to_datetime(df_new['日期'])
    df_new = df_new.sort_values(by='日期')
    df_new = df_new[['日期', '开盘', '收盘', '最高', '最低', '成交量']]

    if os.path.exists(file_path):
        try:
            df_old = pd.read_excel(file_path, sheet_name='日线')
            df_old['日期'] = pd.to_datetime(df_old['日期'])
            # 合并去重，保留新数据
            df_combined = pd.concat([df_old, df_new]).drop_duplicates(subset=['日期'], keep='last')
        except Exception:
            df_combined = df_new
    else:
        df_combined = df_new

    df_combined = df_combined.sort_values('日期').set_index('日期')

    # 计算日线指标
    df_combined = calculate_dkx_logic(df_combined.reset_index()).set_index('日期')
    df_combined = calculate_macd(df_combined)
    df_combined = calculate_ma(df_combined)
    df_combined = calculate_volume_ratio(df_combined)

    # 生成周线、月线
    df_weekly = resample_data(df_combined, 'W-FRI')
    df_monthly = resample_data(df_combined, 'ME')

    # 写入Excel
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df_combined.reset_index().to_excel(writer, sheet_name='日线', index=False)
        df_weekly.reset_index().to_excel(writer, sheet_name='周线', index=False)
        df_monthly.reset_index().to_excel(writer, sheet_name='月线', index=False)

    return True

# ================= 选股逻辑 =================

def check_dkx_cross(df, mode="已经上穿", limit=0.05):
    """检查DKX金叉状态"""
    if len(df) < 2: return False
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    if pd.isna(last['DKX']) or pd.isna(last['MADKX']) or pd.isna(prev['DKX']) or pd.isna(prev['MADKX']):
        return False

    raw_diff = last['DKX'] - last['MADKX']
    diff = abs(raw_diff)

    if mode == "即将上穿":
        return raw_diff < 0 and diff <= limit
    elif mode == "已经上穿":
        # 判断前一日在下方，今日在上方
        return prev['DKX'] <= prev['MADKX'] and raw_diff > 0
    return False

def check_macd_increasing(df, periods=3):
    """检查MACD连续递增"""
    if len(df) < periods + 1: return False
    macd_hist = df['MACD'].dropna()
    if len(macd_hist) < periods + 1: return False
    recent = macd_hist.tail(periods + 1)
    for i in range(len(recent)-1, 0, -1):
        if recent.iloc[i] <= recent.iloc[i-1]:
            return False
    return True

def check_volume_ratio_increase(df, days=3, threshold=1.0):
    """检查量比连续大于1"""
    if len(df) < 6: return False
    recent_ratio = df['量比'].tail(days)
    return all(recent_ratio > threshold)

# ================= 主界面 UI =================

st.title("📈 A股多周期选股器 (回测版)")
st.caption("Tushare Pro 数据源 | 支持导出Excel与增量更新")

with st.sidebar:
    st.header("🔑 数据源设置")
    token_input = st.text_input("Tushare Token", type="password")
    
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
    update_btn = st.button("📥 更新本地数据 (增量)", type="primary", use_container_width=True)
    execute_btn = st.button("🚀 执行选股", type="primary", use_container_width=True)
    reset_btn = st.button("重置", use_container_width=True)

# 数据更新逻辑
if update_btn:
    if not token_input:
        st.warning("请先输入Tushare Token！")
    else:
        stock_list = get_stock_list(token_input)
        if stock_list.empty: st.stop()
        
        # 为了演示，这里只更新前20只股票，实际使用请去掉head
        # stock_list = stock_list.head(20) 
        
        total = len(stock_list)
        st.info(f"开始增量更新 {total} 只股票数据，存入Excel...")
        progress = st.progress(0)
        status = st.empty()
        
        for idx, row in stock_list.iterrows():
            progress.progress((idx+1)/total)
            status.text(f"更新: {row['名称']} ({row['代码']}) - {idx+1}/{total}")
            fetch_and_update_stock(row['代码'], token_input)
            time.sleep(0.3)  # Tushare频率限制
            
        status.text("数据更新完成！")
        progress.empty()
        st.success("所有股票数据已更新至本地Excel！")

# 选股逻辑
if execute_btn:
    if not token_input:
        st.warning("请先输入Tushare Token！")
    else:
        stock_list = get_stock_list(token_input)
        if stock_list.empty: st.stop()
        
        total = len(stock_list)
        st.info(f"正在从本地Excel读取并筛选 {total} 只股票...")
        progress = st.progress(0)
        status = st.empty()
        results = []
        
        for idx, row in stock_list.iterrows():
            code, name = row['代码'], row['名称']
            progress.progress((idx+1)/total)
            status.text(f"筛选: {name} ({code}) - {idx+1}/{total}")
            
            file_path = os.path.join(DATA_DIR, f"{code}.xlsx")
            if not os.path.exists(file_path):
                continue
                
            try:
                df_daily = pd.read_excel(file_path, sheet_name='日线')
                df_weekly = pd.read_excel(file_path, sheet_name='周线')
                df_monthly = pd.read_excel(file_path, sheet_name='月线')
                
                if len(df_daily) < 60 or len(df_weekly) < 30 or len(df_monthly) < 10:
                    continue
                    
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
                    last = df_daily.iloc[-1]
                    prev = df_daily.iloc[-2]
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
