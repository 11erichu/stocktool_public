import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import datetime
import time
import os
# 强制忽略代理
os.environ['NO_PROXY'] = '*'
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

import streamlit as st
import akshare as ak
# ... 其他代码

# 并且在获取数据的地方，增加防报错机制：
@st.cache_data(ttl=3600*4)
def get_stock_list():
    try:
        # 尝试用AkShare获取
        stock_info = ak.stock_zh_a_spot_em()
        return stock_info[['代码', '名称']]
    except Exception as e:
        # 如果失败，尝试用新浪接口替代（或者提示用户重试）
        st.warning(f"从东方财富获取失败，尝试其他源... 错误: {e}")
        try:
            stock_info = ak.stock_zh_a_spot()  # 备用接口
            return stock_info[['代码', '名称']]
        except:
            return pd.DataFrame()
# ================= 页面配置 (手机端适配) =================
st.set_page_config(layout="wide", page_title="A股多周期选股器", page_icon="📈")

st.markdown("""
<style>
    /* 针对手机端优化的CSS */
    @media (max-width: 768px) {
        .block-container { padding: 1rem 0.5rem !important; }
        h1 { font-size: 1.4rem !important; }
        .stCheckbox { font-size: 0.9rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# ================= 核心指标计算函数 =================

def calculate_dkx(df, m=10):
    """计算DKX和MADKX"""
    mid = (3 * df['收盘'] + df['最低'] + df['开盘'] + df['最高']) / 6
    weights = list(range(20, 0, -1)) # 20,19,...,1
    dkx = sum(mid.shift(i) * weights[i-1] for i in range(1, 21)) / 210
    madkx = dkx.rolling(m).mean()
    return dkx, madkx

def calculate_macd(df, short=12, long=26, mid=9):
    """计算MACD柱状线"""
    ema_short = df['收盘'].ewm(span=short, adjust=False).mean()
    ema_long = df['收盘'].ewm(span=long, adjust=False).mean()
    dif = ema_short - ema_long
    dea = dif.ewm(span=mid, adjust=False).mean()
    macd_hist = (dif - dea) * 2
    return macd_hist

def check_dkx_golden_cross(df):
    """判断DKX金叉：前一日DKX <= MADKX，当日DKX > MADKX"""
    if len(df) < 2: return False
    prev_dkx, prev_madkx = df['DKX'].iloc[-2], df['MADKX'].iloc[-2]
    curr_dkx, curr_madkx = df['DKX'].iloc[-1], df['MADKX'].iloc[-1]
    if pd.isna(prev_dkx) or pd.isna(prev_madkx) or pd.isna(curr_dkx) or pd.isna(curr_madkx):
        return False
    return (prev_dkx <= prev_madkx) and (curr_dkx > curr_madkx)

def check_macd_increasing(df, periods=3):
    """检查MACD柱状线连续N个周期递增"""
    if len(df) < periods + 1: return False
    macd_hist = df['MACD'].dropna()
    if len(macd_hist) < periods + 1: return False
    # 检查最近 periods+1 个点是否严格递增
    recent = macd_hist.tail(periods + 1)
    for i in range(len(recent)-1, 0, -1):
        if recent.iloc[i] <= recent.iloc[i-1]:
            return False
    return True

def check_volume_ratio(df, days=3, threshold=1.0):
    """检查量比连续N天大于阈值"""
    if len(df) < 6: return False
    # 量比 = 当日成交量 / 过去5日平均成交量 (不包含当日)
    vol_ma5 = df['成交量'].shift(1).rolling(5).mean()
    volume_ratio = df['成交量'] / vol_ma5
    if volume_ratio.isna().all(): return False
    recent_ratio = volume_ratio.tail(days)
    return all(recent_ratio > threshold)

# ================= 数据获取与处理 =================

@st.cache_data(ttl=3600*4) # 缓存4小时

def get_stock_list():
    """获取全A股列表（加入伪装和备用接口）"""
    import requests
    # 伪装成正常的浏览器访问，防止被识别为爬虫
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    }
    
    # 尝试1：使用新浪财经接口（比东方财富更宽容）
    try:
        st.info("正在尝试从新浪财经获取股票列表...")
        stock_info = ak.stock_zh_a_spot()
        if not stock_info.empty:
            # 新浪接口返回的列名可能不同，做个适配
            stock_info = stock_info.rename(columns={'symbol': '代码', 'name': '名称'})
            return stock_info[['代码', '名称']]
    except Exception as e:
        st.warning(f"新浪接口失败，尝试备用接口... 错误: {e}")

    # 尝试2：使用东方财富接口（加上伪装和延时重试）
    try:
        st.info("正在尝试从东方财富获取股票列表...")
        # 强制休眠2秒，降低请求频率
        time.sleep(2) 
        # 注意：AkShare的东方财富接口通常无法直接传headers，这里我们通过环境变量或AkShare的底层设置来做
        stock_info = ak.stock_zh_a_spot_em()
        if not stock_info.empty:
            return stock_info[['代码', '名称']]
    except Exception as e:
        st.error("所有数据源均获取失败，可能是你的网络IP被暂时封控了。请等待5-10分钟后重试，或者切换网络（如手机开热点给电脑）。")
        return pd.DataFrame()

@st.cache_data(ttl=3600*4)
def get_stock_history(code):
    """获取单只股票近3年日线数据（前复权）"""
    end_date = datetime.datetime.now().strftime('%Y%m%d')
    start_date = (datetime.datetime.now() - datetime.timedelta(days=3*365)).strftime('%Y%m%d')
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        if df.empty: return None
        # 标准化列名
        df = df[['日期', '开盘', '收盘', '最高', '最低', '成交量']]
        df['日期'] = pd.to_datetime(df['日期'])
        df.set_index('日期', inplace=True)
        return df
    except Exception:
        return None

def resample_data(df, period='W'):
    """重采样生成周线或月线"""
    resampled = df.resample(period).agg({
        '开盘': 'first',
        '收盘': 'last',
        '最高': 'max',
        '最低': 'min',
        '成交量': 'sum'
    }).dropna()
    return resampled

# ================= UI界面 =================

st.title("📈 A股多周期选股器")
st.caption("日 / 周 / 月线共振筛选 (数据基于前复权)")

with st.sidebar:
    st.header("⚙️ 选股条件")
    st.write("可单独勾选，按已启用条件筛选")
    
    cond1 = st.checkbox("日线多空线金叉 (DKX/MADKX)", value=True)
    cond2 = st.checkbox("周线多空线金叉 (DKX/MADKX)", value=True)
    cond3 = st.checkbox("日线MACD柱递增 (3天)", value=True)
    cond4 = st.checkbox("周线MACD柱递增 (3根)", value=True)
    cond5 = st.checkbox("月线MACD柱递增 (3根)", value=True)
    cond6 = st.checkbox("量比连续大于1 (3天)", value=True)
    
    st.divider()
    test_mode = st.checkbox("⚡ 快速测试模式 (仅扫描前100只)", value=True, help="首次测试建议勾选，全市场扫描需5-10分钟")
    
    execute_btn = st.button("🚀 执行选股", type="primary", use_container_width=True)
    reset_btn = st.button("重置", use_container_width=True)

if execute_btn:
    if not any([cond1, cond2, cond3, cond4, cond5, cond6]):
        st.warning("请至少选择一个选股条件！")
    else:
        with st.spinner("正在获取全A股列表..."):
            stock_list = get_stock_list()
            
        if stock_list.empty:
            st.stop()
            
        if test_mode:
            stock_list = stock_list.head(100)
            
        total_stocks = len(stock_list)
        st.info(f"本次共需扫描 {total_stocks} 只股票，请耐心等待...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []
        
        for idx, row in stock_list.iterrows():
            code = row['代码']
            name = row['名称']
            
            progress_bar.progress((idx + 1) / total_stocks)
            status_text.text(f"正在分析: {name} ({code}) - {idx+1}/{total_stocks}")
            
            # 获取数据
            df_daily = get_stock_history(code)
            if df_daily is None or len(df_daily) < 60: # 数据太少跳过
                continue
                
            # 计算日线指标
            df_daily['DKX'], df_daily['MADKX'] = calculate_dkx(df_daily)
            df_daily['MACD'] = calculate_macd(df_daily)
            
            # 生成周线和月线
            df_weekly = resample_data(df_daily, 'W-FRI')
            df_monthly = resample_data(df_daily, 'ME') # 月末
            
            if len(df_weekly) < 30 or len(df_monthly) < 10:
                continue
                
            df_weekly['DKX'], df_weekly['MADKX'] = calculate_dkx(df_weekly)
            df_weekly['MACD'] = calculate_macd(df_weekly)
            df_monthly['MACD'] = calculate_macd(df_monthly)
            
            # 条件判断
            match = True
            reasons = []
            
            if cond1:
                if check_dkx_golden_cross(df_daily):
                    reasons.append("日线金叉")
                else:
                    match = False
                    
            if match and cond2:
                if check_dkx_golden_cross(df_weekly):
                    reasons.append("周线金叉")
                else:
                    match = False
                    
            if match and cond3:
                if check_macd_increasing(df_daily, 3):
                    reasons.append("日线MACD递增")
                else:
                    match = False
                    
            if match and cond4:
                if check_macd_increasing(df_weekly, 3):
                    reasons.append("周线MACD递增")
                else:
                    match = False
                    
            if match and cond5:
                if check_macd_increasing(df_monthly, 3):
                    reasons.append("月线MACD递增")
                else:
                    match = False
                    
            if match and cond6:
                if check_volume_ratio(df_daily, 3, 1.0):
                    reasons.append("量比>1")
                else:
                    match = False
            
            if match:
                latest_data = df_daily.iloc[-1]
                prev_close = df_daily['收盘'].iloc[-2]
                change_pct = ((latest_data['收盘'] - prev_close) / prev_close) * 100
                
                results.append({
                    '代码': code,
                    '名称': name,
                    '最新价': round(latest_data['收盘'], 2),
                    '涨跌幅': f"{change_pct:+.2f}%",
                    '匹配条件': " · ".join(reasons)
                })
            
            # 稍微休眠，防止请求过快被封IP
            time.sleep(0.1)
            
        status_text.text("扫描完成！")
        progress_bar.empty()
        
        # 展示结果
        st.subheader(f"🎯 选出 {len(results)} 只股票")
        if results:
            df_results = pd.DataFrame(results)
            # 使用卡片形式或数据框展示
            st.dataframe(df_results, use_container_width=True, hide_index=True)
        else:
            st.info("没有找到符合条件的股票，请尝试放宽条件或稍后再试。")

st.divider()
st.caption("提示：本工具数据来源于AkShare，仅供学习参考，不构成投资建议。")
