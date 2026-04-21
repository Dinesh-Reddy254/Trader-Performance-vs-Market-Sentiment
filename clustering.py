import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')


def build_trader_features(hist_df):
    """Build a per-trader feature matrix suitable for clustering."""
    pnl_df = hist_df[hist_df['Closed PnL'] != 0]

    agg = pnl_df.groupby('Account').agg(
        total_trades=('Closed PnL', 'count'),
        total_pnl=('Closed PnL', 'sum'),
        avg_pnl=('Closed PnL', 'mean'),
        pnl_std=('Closed PnL', 'std'),
        avg_size=('Size USD', 'mean'),
        total_volume=('Size USD', 'sum'),
    ).fillna(0)

    agg['win_rate'] = pnl_df.groupby('Account')['Closed PnL'].apply(lambda x: (x > 0).mean())
    agg['long_pct'] = hist_df.groupby('Account')['Direction'].apply(
        lambda x: x.str.contains('Long', case=False, na=False).mean()
    )

    return agg


def run_clustering(trader_df, n_clusters=4):
    """KMeans clustering on standardized trader features."""
    features = ['total_trades', 'avg_pnl', 'pnl_std', 'avg_size', 'win_rate', 'long_pct']
    X = trader_df[features].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    trader_df['Cluster'] = km.fit_predict(X_scaled)

    archetype_map = {
        trader_df.groupby('Cluster')['avg_pnl'].mean().idxmax(): 'High-Performance',
        trader_df.groupby('Cluster')['total_trades'].mean().idxmax(): 'High-Frequency',
        trader_df.groupby('Cluster')['pnl_std'].mean().idxmax(): 'Volatile/Risky',
        trader_df.groupby('Cluster')['win_rate'].mean().idxmin(): 'Struggling',
    }
    trader_df['Archetype'] = trader_df['Cluster'].map(archetype_map).fillna('Moderate')

    print("\n--- Cluster Summary ---")
    print(trader_df.groupby('Archetype')[features].mean().round(3))

    # PCA plot
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    trader_df['pca1'] = coords[:, 0]
    trader_df['pca2'] = coords[:, 1]

    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=trader_df, x='pca1', y='pca2', hue='Archetype', palette='Set2', s=80)
    plt.title('Trader Behavioral Archetypes (PCA View)')
    plt.tight_layout()
    plt.savefig('trader_clusters.png')
    plt.close()
    print("Cluster chart saved to trader_clusters.png")

    joblib.dump(km, 'cluster_model.pkl')
    trader_df.to_csv('trader_segments.csv')
    print("Trader segments saved to trader_segments.csv")
    return trader_df


if __name__ == '__main__':
    hist_df = pd.read_csv('historical_data.csv')
    hist_df['date'] = pd.to_datetime(hist_df['Timestamp IST'], dayfirst=True, errors='coerce').dt.date
    hist_df['Closed PnL'] = pd.to_numeric(hist_df['Closed PnL'], errors='coerce').fillna(0)
    hist_df['Size USD'] = pd.to_numeric(hist_df['Size USD'], errors='coerce').fillna(0)

    trader_df = build_trader_features(hist_df)
    trader_df = run_clustering(trader_df)
