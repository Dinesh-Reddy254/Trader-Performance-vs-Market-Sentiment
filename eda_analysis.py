import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda():
    # just setting up the paths
    sentiment_file = 'fear_greed_index.csv'
    trade_history_file = 'historical_data.csv'
    
    if not os.path.exists(sentiment_file) or not os.path.exists(trade_history_file):
        print("Hey, looks like the CSV files are missing. Make sure they're in the same folder!")
        return

    print("Loading up the datasets...")
    fg_df = pd.read_csv(sentiment_file)
    trades_df = pd.read_csv(trade_history_file)
    
    print("Fixing timestamps and aligning dates...")
    # convert dates so we can merge them properly
    fg_df['date'] = pd.to_datetime(fg_df['date']).dt.date
    trades_df['date'] = pd.to_datetime(trades_df['Timestamp IST'], dayfirst=True, errors='coerce').dt.date
    
    # clean up numeric columns just to be safe
    trades_df['Closed PnL'] = pd.to_numeric(trades_df['Closed PnL'], errors='coerce').fillna(0)
    trades_df['Size USD'] = pd.to_numeric(trades_df['Size USD'], errors='coerce').fillna(0)
    
    # join everything together based on the trading date
    merged_data = pd.merge(trades_df, fg_df, on='date', how='left')
    
    print("Crunching the numbers...")
    
    # how often are we actually winning?
    trades_with_pnl = merged_data[merged_data['Closed PnL'] != 0]
    win_rate = (trades_with_pnl['Closed PnL'] > 0).mean() * 100
    
    # avg trade size
    avg_size = merged_data['Size USD'].mean()
    
    # roughly how many trades a day?
    daily_trade_count = merged_data.groupby('date').size()
    
    # long/short split
    long_count = merged_data['Direction'].str.contains('Long', case=False, na=False).sum()
    short_count = merged_data['Direction'].str.contains('Short', case=False, na=False).sum()
    ls_ratio = long_count / short_count if short_count > 0 else 0

    print(f"-> Win Rate: {win_rate:.2f}%")
    print(f"-> Avg Trade Size: ${avg_size:,.2f}")
    print(f"-> Trades per Day: ~{int(daily_trade_count.mean())}")
    print(f"-> Long/Short Ratio: {ls_ratio:.2f} ({long_count} longs vs {short_count} shorts)")
    
    print("\nGenerating some charts now...")
    sns.set_theme(style="whitegrid")
    
    # Chart 1: Daily PnL trend
    daily_pnl = merged_data.groupby('date')['Closed PnL'].sum().reset_index()
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=daily_pnl, x='date', y='Closed PnL', color='dodgerblue')
    plt.title('Daily Realized PnL')
    plt.xlabel('Date')
    plt.ylabel('PnL (USD)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('daily_pnl.png')
    plt.close()
    
    # Chart 2: PnL vs Market Sentiment (Fear/Greed)
    daily_combined = pd.merge(daily_pnl, fg_df, on='date', how='inner')
    plt.figure(figsize=(9, 5))
    sns.scatterplot(data=daily_combined, x='value', y='Closed PnL', hue='classification', palette='coolwarm')
    plt.title('Does Sentiment Affect PnL?')
    plt.xlabel('Fear & Greed Index (0=Fear, 100=Greed)')
    plt.ylabel('Daily PnL (USD)')
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('pnl_vs_sentiment.png')
    plt.close()
    
    # Chart 3: Longs vs Shorts
    plt.figure(figsize=(5, 4))
    import warnings
    warnings.filterwarnings("ignore") # hide silly seaborn warnings
    sns.barplot(x=['Longs', 'Shorts'], y=[long_count, short_count], hue=['Longs', 'Shorts'], palette=['#2ecc71', '#e74c3c'], dodge=False, legend=False)
    plt.title('Long vs Short Breakdown')
    plt.ylabel('Number of Trades')
    plt.tight_layout()
    plt.savefig('long_vs_short.png')
    plt.close()
    
    # save a quick summary table
    summary_df = pd.DataFrame({
        'Metric': ['Win Rate', 'Avg Trade Size', 'Avg Trades/Day', 'L/S Ratio', 'Total PnL'],
        'Value': [f"{win_rate:.2f}%", f"${avg_size:,.2f}", f"{int(daily_trade_count.mean())}", f"{ls_ratio:.2f}", f"${merged_data['Closed PnL'].sum():,.2f}"]
    })
    summary_df.to_csv('summary_metrics.csv', index=False)
    print("All done! Check the folder for the new PNG charts and the CSV summary.")

if __name__ == '__main__':
    run_eda()
