import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
import datetime
import time

# ================= 页面配置 (手机端深度优化) =================
st.set_page_config(layout="wide", page_title="A股多周期选股器", page_icon="📈")

st.markdown("""
<style>
    /* 针对手机端优化的CSS */
    @media (max-width: 768px) {
        .block-container { padding: 1rem 0.5rem !important; }
        h1 { font-size: 1.4rem !important; }
        .stTabs [data-baseweb="tab"] { font-size: 0.85rem !important; padding: 0.5rem !important; }
        .stCheckbox { font-size: 0.9rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# ================= 核心指标计算函数 =================

def calculate_dkx(df, m=10):
    """计算DKX和MADKX"""
    mid = (3 * df['收盘'] + df['最低'] + df['开盘'] + df['最高']) / 6
    weights = list(range(20, 0, -1))
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
    """判断DKX金叉"""
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
    recent = macd_hist.tail(periods + 1)
    for i in range(len(recent)-1, 0, -1):
        if recent.iloc[i] <= recent.iloc[i-1]:
            return False
    return True

def check_volume_ratio(df, days=3, threshold=1.0):
    """检查量比连续N天大于阈值"""
    if len(df) < 6: return False
    vol_ma5 = df['成交量'].shift(1).rolling(5).mean()
    volume_ratio = df['成交量'] / vol_ma5
    if volume_ratio.isna().all(): return False
    recent_ratio = volume_ratio.tail(days)
    return all(recent_ratio > threshold)

def resample_data(df, period='W'):
    """重采样生成周线或月线"""
    resampled = df.resample(period).agg({
        '开盘': 'first', '收盘': 'last', '最高': 'max', '最低': 'min', '成交量': 'sum'
    }).dropna()
    return resampled

# ================= 数据获取层 (Tushare Pro) =================

@st.cache_data(ttl=3600*24)  # 缓存24小时，避免重复请求
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

@st.cache_data(ttl=3600*12)  # 缓存12小时
def get_stock_history(code, token):
    """获取单只股票近3年日线数据（前复权）"""
    pro = ts.pro_api(token)
    # 自动判断交易所后缀
    ts_code = code + '.SZ' if code.startswith(('0', '3')) else code + '.SH'
    
    end_date = datetime.datetime.now().strftime('%Y%m%d')
    start_date = (datetime.datetime.now() - datetime.timedelta(days=3*365)).strftime('%Y%m%d')
    
    try:
        # 注意：获取前复权数据必须使用 pro_bar 接口
        df = ts.pro_bar(ts_code=ts_code, start_date=start_date, end_date=end_date, adj='qfq')
        if df is None or df.empty:
            return None
        # 列名映射与格式化
        df = df.rename(columns={'trade_date': '日期', 'open': '开盘', 'close': '收盘', 'high': '最高', 'low': '最低', 'vol': '成交量'})
        df['日期'] = pd.to_datetime(df['日期'])
        df.set_index('日期', inplace=True)
        df = df.sort_index()  # Tushare默认是倒序，必须正序排列
        return df[['开盘', '收盘', '最高', '最低', '成交量']]
    except Exception:
        return None

# ================= 主界面 UI =================

st.title("📈 A股多周期选股器")
st.caption("日 / 周 / 月线共振筛选 (Tushare Pro 数据源)")

# 侧边栏：Token 和 条件配置
with st.sidebar:
    st.header("🔑 数据源设置")
    token_input = st.text_input("Tushare Token", type="password", help="在 tushare.pro 个人中心获取")
    
    st.divider()
    st.header("⚙️ 选股条件")
    
    # 使用 Tabs 对条件进行分组，手机端操作更清爽
    tab1, tab2, tab3 = st.tabs(["📊 多空线", "📈 MACD", "⚖️ 量价"])
    
    with tab1:
        cond1 = st.checkbox("日线多空线金叉 (DKX/MADKX)", value=True)
        cond2 = st.checkbox("周线多空线金叉 (DKX/MADKX)", value=True)
    
    with tab2:
        cond3 = st.checkbox("日线MACD柱递增 (3天)", value=True)
        cond4 = st.checkbox("周线MACD柱递增 (3根)", value=True)
        cond5 = st.checkbox("月线MACD柱递增 (3根)", value=False)
        
    with tab3:
        cond6 = st.checkbox("量比连续大于1 (3天)", value=True)
    
    st.divider()
    test_mode = st.checkbox("⚡ 快速测试模式 (仅扫描前100只)", value=True, help="首次测试建议勾选，全市场扫描需5-10分钟")
    
    execute_btn = st.button("🚀 执行选股", type="primary", use_container_width=True)
    reset_btn = st.button("重置", use_container_width=True)

# 主逻辑执行
if execute_btn:
    if not token_input:
        st.warning("请先在左侧输入你的 Tushare Token！")
    elif not any([cond1, cond2, cond3, cond4, cond5, cond6]):
        st.warning("请至少选择一个选股条件！")
    else:
        with st.spinner("正在获取全A股列表..."):
            stock_list = get_stock_list(token_input)
            
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
            
            # 获取数据 (从本地缓存或Tushare)
            df_daily = get_stock_history(code, token_input)
            if df_daily is None or len(df_daily) < 60:
                continue
                
            # 计算日线指标
            df_daily['DKX'], df_daily['MADKX'] = calculate_dkx(df_daily)
            df_daily['MACD'] = calculate_macd(df_daily)
            
            # 生成周线和月线
            df_weekly = resample_data(df_daily, 'W-FRI')
            df_monthly = resample_data(df_daily, 'ME')
            
            if len(df_weekly) < 30 or len(df_monthly) < 10:
                continue
                
            df_weekly['DKX'], df_weekly['MADKX'] = calculate_dkx(df_weekly)
            df_weekly['MACD'] = calculate_macd(df_weekly)
            df_monthly['MACD'] = calculate_macd(df_monthly)
            
            # 条件判断
            match = True
            reasons = []
            
            if cond1:
                if check_dkx_golden_cross(df_daily): reasons.append("日线金叉")
                else: match = False
                    
            if match and cond2:
                if check_dkx_golden_cross(df_weekly): reasons.append("周线金叉")
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
                if check_volume_ratio(df_daily, 3, 1.0): reasons.append("量比>1")
                else: match = False
            
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
            
            # 关键：休眠0.3秒，防止Tushare触发频率限制
            time.sleep(0.3)
            
        status_text.text("扫描完成！")
        progress_bar.empty()
        
        # 展示结果
        st.subheader(f"🎯 选出 {len(results)} 只股票")
        if results:
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True, hide_index=True)
        else:
            st.info("没有找到符合条件的股票，请尝试放宽条件或稍后再试。")

st.divider()
st.caption("提示：本工具数据来源于Tushare Pro，仅供学习参考，不构成投资建议。")
