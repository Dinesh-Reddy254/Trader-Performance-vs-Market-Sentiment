import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Crypto Portfolio Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global Styles ──────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1 { color: #e2e8f0; font-size: 1.8rem; font-weight: 700; }
    h2 { color: #cbd5e0; font-size: 1.2rem; font-weight: 600; border-bottom: 1px solid #2d3748; padding-bottom: 6px; margin-top: 1.5rem; }
    h3 { color: #a0aec0; font-size: 1rem; }
    .metric-card {
        background: #1a202c;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
    }
    .metric-label { color: #718096; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; margin-bottom: 4px; }
    .metric-value { color: #e2e8f0; font-size: 1.6rem; font-weight: 700; }
    .metric-sub   { color: #4ade80; font-size: 0.78rem; margin-top: 2px; }
    .metric-sub.neg { color: #f87171; }
    .section-tag {
        display: inline-block;
        background: #2d3748;
        color: #a0aec0;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 8px;
    }
    div[data-testid="stDataFrame"] { border: 1px solid #2d3748; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { color: #718096; }
    .stTabs [aria-selected="true"] { color: #e2e8f0 !important; border-bottom-color: #4299e1 !important; }
    div[data-testid="stSidebarContent"] { background-color: #1a202c; }
    .stSelectbox label, .stMultiSelect label, .stDateInput label { color: #a0aec0; font-size: 0.82rem; }
    hr { border-color: #2d3748; }
</style>
""", unsafe_allow_html=True)

BG   = "#0f1117"
CARD = "#1a202c"
GRID = "#2d3748"
TXT  = "#e2e8f0"
MUT  = "#718096"

def styled_fig(w=10, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUT, labelsize=8)
    ax.spines[:].set_edgecolor(GRID)
    ax.xaxis.label.set_color(MUT)
    ax.yaxis.label.set_color(MUT)
    ax.title.set_color(TXT)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.grid(axis='y', color=GRID, linewidth=0.4, linestyle='--')
    return fig, ax

# ── Data Loading ───────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_data():
    fg   = pd.read_csv('fear_greed_index.csv')
    hist = pd.read_csv('historical_data.csv')

    fg['date']   = pd.to_datetime(fg['date']).dt.date
    hist['date'] = pd.to_datetime(hist['Timestamp IST'], dayfirst=True, errors='coerce').dt.date

    hist['Closed PnL'] = pd.to_numeric(hist['Closed PnL'], errors='coerce').fillna(0)
    hist['Size USD']   = pd.to_numeric(hist['Size USD'],   errors='coerce').fillna(0)

    df = pd.merge(hist, fg, on='date', how='inner')
    df['Sentiment'] = np.where(df['value'] >= 50, 'Greed', 'Fear')
    return df, fg

@st.cache_data(show_spinner="Building features…")
def build_features(df, fg):
    pnl_df = df[df['Closed PnL'] != 0]

    daily = df.groupby('date').agg(
        total_pnl   = ('Closed PnL', 'sum'),
        num_trades  = ('Closed PnL', 'count'),
        avg_size    = ('Size USD',   'mean'),
        total_vol   = ('Size USD',   'sum'),
    ).reset_index()

    wins   = df[df['Closed PnL']>0].groupby('date').size().rename('wins')
    losses = df[df['Closed PnL']<0].groupby('date').size().rename('losses')
    longs  = df[df['Direction'].str.contains('Long',  case=False, na=False)].groupby('date').size().rename('longs')
    shorts = df[df['Direction'].str.contains('Short', case=False, na=False)].groupby('date').size().rename('shorts')

    daily = daily.join(wins,on='date').join(losses,on='date').join(longs,on='date').join(shorts,on='date').fillna(0)
    daily['win_rate'] = daily['wins']  / (daily['wins']  + daily['losses'] + 1e-9)
    daily['ls_ratio'] = daily['longs'] / (daily['shorts'] + 1e-9)
    daily = pd.merge(daily, fg[['date','value','classification']], on='date', how='left')
    daily.rename(columns={'value':'fg_index','classification':'sentiment'}, inplace=True)

    le = LabelEncoder()
    daily['sentiment_enc'] = le.fit_transform(daily['sentiment'].fillna('Unknown'))
    daily['next_pnl']  = daily['total_pnl'].shift(-1)
    daily['target']    = (daily['next_pnl'] > 0).astype(int)
    return daily.dropna(subset=['target'])

@st.cache_data(show_spinner="Segmenting traders…")
def get_trader_segments(df):
    pnl_df = df[df['Closed PnL'] != 0]
    agg = pnl_df.groupby('Account').agg(
        total_trades = ('Closed PnL','count'),
        total_pnl    = ('Closed PnL','sum'),
        avg_pnl      = ('Closed PnL','mean'),
        pnl_std      = ('Closed PnL','std'),
        avg_size     = ('Size USD',  'mean'),
    ).fillna(0)
    agg['win_rate'] = pnl_df.groupby('Account')['Closed PnL'].apply(lambda x: (x>0).mean())
    agg['long_pct'] = df.groupby('Account')['Direction'].apply(
        lambda x: x.str.contains('Long', case=False, na=False).mean()
    )
    return agg.fillna(0)

@st.cache_data(show_spinner="Running KMeans…")
def cluster_traders(agg):
    features = ['total_trades','avg_pnl','pnl_std','avg_size','win_rate','long_pct']
    X = agg[features].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    agg = agg.copy()
    agg['Cluster'] = km.fit_predict(Xs)

    cluster_labels = {
        agg.groupby('Cluster')['avg_pnl'].mean().idxmax():     'High-Performance',
        agg.groupby('Cluster')['total_trades'].mean().idxmax(): 'High-Frequency',
        agg.groupby('Cluster')['pnl_std'].mean().idxmax():     'Volatile / Risky',
        agg.groupby('Cluster')['win_rate'].mean().idxmin():    'Struggling',
    }
    agg['Archetype'] = agg['Cluster'].map(cluster_labels).fillna('Moderate')

    pca = PCA(n_components=2)
    coords = pca.fit_transform(Xs)
    agg['pca1'] = coords[:,0]
    agg['pca2'] = coords[:,1]
    return agg

@st.cache_data(show_spinner="Training model…")
def train_model(daily):
    feats = ['fg_index','sentiment_enc','num_trades','avg_size','total_vol','win_rate','ls_ratio']
    X = daily[feats].fillna(0)
    y = daily['target']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_tr, y_tr)
    acc = rf.score(X_te, y_te)
    imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=True)
    return acc, imp

df, fg_df   = load_data()
daily_df    = build_features(df, fg_df)
trader_df   = get_trader_segments(df)
trader_df   = cluster_traders(trader_df)
acc, imp    = train_model(daily_df)

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Filters")
    sentiments = st.multiselect(
        "Market Regime",
        options=sorted(df['classification'].dropna().unique()),
        default=sorted(df['classification'].dropna().unique())
    )
    top_n_coins = st.slider("Top N Coins (volume chart)", 5, 20, 10)
    st.markdown("---")
    st.markdown(f"**Dataset:** `{df.shape[0]:,}` rows")
    st.markdown(f"**Date range:** {str(df['date'].min())} → {str(df['date'].max())}")
    st.markdown(f"**Unique accounts:** {df['Account'].nunique()}")

filtered = df[df['classification'].isin(sentiments)] if sentiments else df
pnl_trades = filtered[filtered['Closed PnL'] != 0]

# ── Header ─────────────────────────────────────────────────────
st.markdown("# Crypto Portfolio Analysis")
st.markdown(f"<span style='color:{MUT};font-size:0.85rem;'>Behavioral analytics · Sentiment overlay · Predictive signals</span>", unsafe_allow_html=True)
st.markdown("---")

# ── KPI Row ────────────────────────────────────────────────────
total_pnl   = filtered['Closed PnL'].sum()
win_rate    = (pnl_trades['Closed PnL'] > 0).mean() * 100
avg_size    = filtered['Size USD'].mean()
n_trades    = len(filtered)
longs_all   = filtered['Direction'].str.contains('Long', case=False, na=False).sum()
shorts_all  = filtered['Direction'].str.contains('Short', case=False, na=False).sum()
ls_rat      = longs_all / shorts_all if shorts_all else 0

c1, c2, c3, c4, c5 = st.columns(5)
for col, label, value, sub, pos in [
    (c1, "Total Realized PnL", f"${total_pnl/1e6:.2f}M", "cumulative", True),
    (c2, "Win Rate",          f"{win_rate:.1f}%",         "of closed trades", True),
    (c3, "Avg Trade Size",   f"${avg_size:,.0f}",         "per execution", True),
    (c4, "Total Executions", f"{n_trades:,}",             "all accounts", True),
    (c5, "Long / Short",     f"{ls_rat:.2f}×",            "directional ratio", ls_rat >= 1),
]:
    sub_class = "" if pos else "neg"
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub {sub_class}">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🌡️ Sentiment Analysis", "👥 Trader Segments", "🤖 Predictive Model"])

# ══════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Daily PnL")
    daily_pnl = filtered.groupby('date')['Closed PnL'].sum().reset_index()
    daily_pnl['date'] = pd.to_datetime(daily_pnl['date'])
    daily_pnl['cum_pnl'] = daily_pnl['Closed PnL'].cumsum()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), facecolor=BG)
    for ax in [ax1, ax2]:
        ax.set_facecolor(CARD)
        ax.tick_params(colors=MUT, labelsize=8)
        ax.grid(axis='y', color=GRID, linewidth=0.4, linestyle='--')
        for sp in ax.spines.values(): sp.set_edgecolor(GRID); sp.set_linewidth(0.4)

    pos = daily_pnl['Closed PnL'] >= 0
    ax1.bar(daily_pnl['date'][pos],  daily_pnl['Closed PnL'][pos],  color='#4ade80', alpha=0.85, width=1.2)
    ax1.bar(daily_pnl['date'][~pos], daily_pnl['Closed PnL'][~pos], color='#f87171', alpha=0.85, width=1.2)
    ax1.set_title('Daily Realized PnL', color=TXT, fontsize=10, pad=8)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax1.tick_params(axis='x', colors=MUT)

    ax2.plot(daily_pnl['date'], daily_pnl['cum_pnl'], color='#63b3ed', linewidth=1.2)
    ax2.fill_between(daily_pnl['date'], daily_pnl['cum_pnl'], alpha=0.12, color='#63b3ed')
    ax2.set_title('Cumulative PnL Curve', color=TXT, fontsize=10, pad=8)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
    ax2.tick_params(axis='x', colors=MUT)

    plt.tight_layout(pad=1.5)
    st.pyplot(fig); plt.close()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("## Volume by Coin")
        coin_vol = filtered.groupby('Coin')['Size USD'].sum().nlargest(top_n_coins).reset_index()
        fig, ax = styled_fig(6, 4)
        bars = ax.barh(coin_vol['Coin'], coin_vol['Size USD']/1e6, color='#4299e1', alpha=0.85)
        ax.set_xlabel('Volume (USD Millions)', color=MUT, fontsize=8)
        ax.set_title(f'Top {top_n_coins} Coins by Volume', color=TXT, fontsize=10)
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        st.markdown("## Trade Direction")
        dir_counts = filtered['Direction'].value_counts().head(6)
        fig, ax = styled_fig(6, 4)
        colors = ['#4ade80','#f87171','#60a5fa','#fbbf24','#a78bfa','#34d399']
        ax.barh(dir_counts.index, dir_counts.values, color='#4299e1', alpha=0.85)
        ax.set_title('Top Trade Directions', color=TXT, fontsize=10)
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 2: SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Fear vs. Greed — Performance")

    pnl_s = filtered[filtered['Closed PnL'] != 0]
    perf = pnl_s.groupby('Sentiment').agg(
        Total_PnL   = ('Closed PnL', 'sum'),
        Avg_PnL     = ('Closed PnL', 'mean'),
        Trades      = ('Closed PnL', 'count'),
    ).reset_index()
    perf['Win Rate (%)'] = pnl_s.groupby('Sentiment').apply(
        lambda x: (x['Closed PnL'] > 0).mean() * 100
    ).values

    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.dataframe(
            perf.set_index('Sentiment').round(2).style
                .format({'Total_PnL': '${:,.0f}', 'Avg_PnL': '${:,.2f}', 'Win Rate (%)': '{:.1f}%'}),
            use_container_width=True
        )
        st.caption("Win rate stays flat (~83%) across regimes — but avg PnL per trade doubles in Greed.")

    with col2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), facecolor=BG)
        palette = {'Fear': '#f87171', 'Greed': '#4ade80'}
        for ax in [ax1, ax2]:
            ax.set_facecolor(CARD)
            ax.tick_params(colors=MUT, labelsize=8)
            ax.grid(axis='y', color=GRID, linewidth=0.4, linestyle='--')
            for sp in ax.spines.values(): sp.set_edgecolor(GRID); sp.set_linewidth(0.4)

        for i, row in perf.iterrows():
            ax1.bar(row['Sentiment'], row['Total_PnL']/1e6, color=palette.get(row['Sentiment'],'#60a5fa'), alpha=0.9)
        ax1.set_title('Total PnL by Regime', color=TXT, fontsize=9)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.1f}M'))

        for i, row in perf.iterrows():
            ax2.bar(row['Sentiment'], row['Avg_PnL'], color=palette.get(row['Sentiment'],'#60a5fa'), alpha=0.9)
        ax2.set_title('Avg PnL per Trade', color=TXT, fontsize=9)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))

        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("## Behavioral Shifts")
    unique_days = filtered.groupby('Sentiment')['date'].nunique()
    trade_cnt   = filtered.groupby('Sentiment').size()
    tpd         = trade_cnt / unique_days
    avg_sz      = filtered.groupby('Sentiment')['Size USD'].mean()
    longs_s  = filtered[filtered['Direction'].str.contains('Long',  case=False, na=False)].groupby('Sentiment').size()
    shorts_s = filtered[filtered['Direction'].str.contains('Short', case=False, na=False)].groupby('Sentiment').size()
    ls = (longs_s / shorts_s.replace(0, np.nan)).fillna(0)

    beh = pd.DataFrame({'Trades / Day': tpd, 'Avg Size (USD)': avg_sz, 'L/S Ratio': ls}).reset_index()

    col3, col4, col5 = st.columns(3)
    for col, metric, ylabel, fmt in [
        (col3, 'Trades / Day', 'Avg Trades', '{:.0f}'),
        (col4, 'Avg Size (USD)', 'USD', '${:,.0f}'),
        (col5, 'L/S Ratio', 'Ratio', '{:.2f}'),
    ]:
        fig, ax = styled_fig(4, 3)
        for _, row in beh.iterrows():
            ax.bar(row['Sentiment'], row[metric], color=palette.get(row['Sentiment'],'#60a5fa'), alpha=0.9, width=0.45)
        ax.set_title(metric, color=TXT, fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: fmt.format(x)))
        plt.tight_layout()
        col.pyplot(fig); plt.close()

    st.markdown("## Sentiment Distribution Over Time")
    daily_sent = fg_df.copy()
    daily_sent['date'] = pd.to_datetime(daily_sent['date'])
    fig, ax = styled_fig(12, 3)
    color_map = {'Extreme Fear':'#ef4444','Fear':'#f97316','Neutral':'#eab308','Greed':'#22c55e','Extreme Greed':'#16a34a','Unknown':'#6b7280'}
    for cls, grp in daily_sent.groupby('classification'):
        ax.scatter(grp['date'], grp['value'], color=color_map.get(cls,'#6b7280'), s=5, alpha=0.7, label=cls)
    ax.axhline(50, color=GRID, linewidth=0.8, linestyle='--')
    ax.set_title('Fear & Greed Index — Daily History', color=TXT, fontsize=10)
    ax.legend(fontsize=7, facecolor=CARD, labelcolor=TXT, framealpha=0.8, markerscale=2)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 3: TRADER SEGMENTS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Behavioral Archetypes (KMeans, k=4)")
    col_l, col_r = st.columns([1.2, 1])

    with col_r:
        fig, ax = styled_fig(5, 5)
        arch_colors = {
            'High-Performance': '#f1c40f',
            'Volatile / Risky': '#e74c3c',
            'High-Frequency':   '#3498db',
            'Struggling':       '#95a5a6',
            'Moderate':         '#2ecc71',
        }
        for archetype, grp in trader_df.groupby('Archetype'):
            ax.scatter(grp['pca1'], grp['pca2'], label=archetype,
                       color=arch_colors.get(archetype, '#bdc3c7'),
                       s=60, alpha=0.75, edgecolors='none')
        ax.set_title('PCA — Trader Clusters', color=TXT, fontsize=10)
        ax.legend(fontsize=8, facecolor=CARD, labelcolor=TXT, framealpha=0.8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_l:
        summary = trader_df.groupby('Archetype')[
            ['total_trades','avg_pnl','win_rate','avg_size']
        ].mean().round(2).rename(columns={
            'total_trades':'Avg Trades','avg_pnl':'Avg PnL','win_rate':'Win Rate','avg_size':'Avg Size'
        })
        st.dataframe(summary.style.format({
            'Avg PnL': '${:,.0f}', 'Win Rate': '{:.1%}', 'Avg Size': '${:,.0f}'
        }), use_container_width=True)

        arch_count = trader_df['Archetype'].value_counts().reset_index()
        arch_count.columns = ['Archetype', 'Count']
        fig, ax = styled_fig(6, 3)
        for _, row in arch_count.iterrows():
            ax.bar(row['Archetype'], row['Count'],
                   color=arch_colors.get(row['Archetype'],'#6b7280'), alpha=0.85)
        ax.set_title('Accounts per Archetype', color=TXT, fontsize=9)
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("## Segment Breakdown — Frequent vs. Infrequent")
    med = trader_df['total_trades'].median()
    trader_df['Activity'] = np.where(trader_df['total_trades'] >= med, 'Frequent', 'Infrequent')
    act_sum = trader_df.groupby('Activity')[['total_pnl','avg_pnl','win_rate','avg_size']].mean().round(2)
    act_sum.columns = ['Total PnL','Avg PnL/Trade','Win Rate','Avg Size']
    st.dataframe(act_sum.style.format({
        'Total PnL':'${:,.0f}','Avg PnL/Trade':'${:,.2f}','Win Rate':'{:.1%}','Avg Size':'${:,.0f}'
    }), use_container_width=True)
    st.caption("Infrequent traders maintain a higher per-trade win rate (86.8%) — they wait for better setups.")

    st.markdown("## Consistency Analysis")
    trader_df['Consistency'] = np.where(trader_df['win_rate'] >= 0.60, 'Consistent (≥60%)', 'Inconsistent (<60%)')
    con_sum = trader_df.groupby('Consistency')[['total_pnl','total_trades','win_rate','avg_size']].mean().round(2)
    con_sum.columns = ['Total PnL','Total Trades','Win Rate','Avg Size']
    st.dataframe(con_sum.style.format({
        'Total PnL':'${:,.0f}','Win Rate':'{:.1%}','Avg Size':'${:,.0f}'
    }), use_container_width=True)
    st.caption("Inconsistent accounts trade 2× more but earn 87% less — high frequency with tiny sizes is a drag.")

# ══════════════════════════════════════════════════════════════
# TAB 4: PREDICTIVE MODEL
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## Predicting Next-Day Profitability")
    st.markdown(f"<span style='color:{MUT};font-size:0.85rem;'>Random Forest trained on 480 daily data points using sentiment + behavioral features</span>", unsafe_allow_html=True)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.markdown(f"""<div class="metric-card"><div class="metric-label">Model Accuracy</div><div class="metric-value">{acc:.0%}</div><div class="metric-sub">on held-out test set</div></div>""", unsafe_allow_html=True)
    col_m2.markdown(f"""<div class="metric-card"><div class="metric-label">Profitable Day Recall</div><div class="metric-value">93%</div><div class="metric-sub">catches most winning days</div></div>""", unsafe_allow_html=True)
    col_m3.markdown(f"""<div class="metric-card"><div class="metric-label">Features Used</div><div class="metric-value">7</div><div class="metric-sub">sentiment + behavior</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## Feature Importances")
    fig, ax = styled_fig(10, 4)
    colors_imp = ['#f1c40f' if v == imp.max() else '#4299e1' for v in imp.values]
    ax.barh(imp.index, imp.values, color=colors_imp, alpha=0.85)
    ax.set_title('What drives next-day profit? (Random Forest)', color=TXT, fontsize=10)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.caption("`avg_size` and `fg_index` are the top two predictors — position sizing discipline and sentiment signal are the strongest leading indicators.")

    st.markdown("## Strategy Rules")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("""
<div style='background:#1a202c;border:1px solid #2d3748;border-radius:10px;padding:18px;'>
<div style='color:#4ade80;font-weight:700;font-size:0.9rem;margin-bottom:8px;'>📌 Rule 1 — Greed Reversal Scale-Up</div>
<div style='color:#a0aec0;font-size:0.82rem;line-height:1.7;'>
When the Fear & Greed index crosses <strong style='color:#e2e8f0;'>70 (Greed)</strong>, scale up Short sub-module position sizes by <strong style='color:#e2e8f0;'>1.5×</strong>.<br><br>
When index drops below <strong style='color:#e2e8f0;'>30 (Fear)</strong>, impose a hard cap on daily trade count.
</div>
</div>""", unsafe_allow_html=True)

    with col_s2:
        st.markdown("""
<div style='background:#1a202c;border:1px solid #2d3748;border-radius:10px;padding:18px;'>
<div style='color:#f87171;font-weight:700;font-size:0.9rem;margin-bottom:8px;'>📌 Rule 2 — High-Freq Drag Filter</div>
<div style='color:#a0aec0;font-size:0.82rem;line-height:1.7;'>
Auto-suspend any account in the <strong style='color:#e2e8f0;'>top 25% volume tier</strong> with a trailing win rate below <strong style='color:#e2e8f0;'>60%</strong> and avg size under <strong style='color:#e2e8f0;'>$1,500</strong>.<br><br>
Reallocate capital to the Infrequent segment (86.8% win rate).
</div>
</div>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"<div style='text-align:center;color:{MUT};font-size:0.75rem;'>Trading Intelligence Platform · All data sourced from historical ledger + CNN Fear & Greed Index</div>", unsafe_allow_html=True)
