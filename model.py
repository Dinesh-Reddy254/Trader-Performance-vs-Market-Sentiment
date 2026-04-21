import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

def build_features(hist_df, fg_df):
    """Build daily feature set merging trade behavior with sentiment."""

    # daily aggregates per day
    daily = hist_df.groupby('date').agg(
        total_pnl=('Closed PnL', 'sum'),
        num_trades=('Closed PnL', 'count'),
        avg_size=('Size USD', 'mean'),
        total_volume=('Size USD', 'sum'),
    ).reset_index()

    wins = hist_df[hist_df['Closed PnL'] > 0].groupby('date').size().rename('wins')
    losses = hist_df[hist_df['Closed PnL'] < 0].groupby('date').size().rename('losses')
    longs = hist_df[hist_df['Direction'].str.contains('Long', case=False, na=False)].groupby('date').size().rename('longs')
    shorts = hist_df[hist_df['Direction'].str.contains('Short', case=False, na=False)].groupby('date').size().rename('shorts')

    daily = daily.join(wins, on='date').join(losses, on='date').join(longs, on='date').join(shorts, on='date')
    daily = daily.fillna(0)

    daily['win_rate'] = daily['wins'] / (daily['wins'] + daily['losses'] + 1e-9)
    daily['ls_ratio'] = daily['longs'] / (daily['shorts'] + 1e-9)

    # merge sentiment
    daily = pd.merge(daily, fg_df[['date', 'value', 'classification']], on='date', how='left')
    daily.rename(columns={'value': 'fg_index', 'classification': 'sentiment'}, inplace=True)

    # encode sentiment
    le = LabelEncoder()
    daily['sentiment_enc'] = le.fit_transform(daily['sentiment'].fillna('Unknown'))

    # target: next-day PnL is positive (1) or not (0)
    daily['next_day_pnl'] = daily['total_pnl'].shift(-1)
    daily['target'] = (daily['next_day_pnl'] > 0).astype(int)

    # pnl volatility bucket (low / medium / high based on rolling std)
    daily['pnl_std'] = daily['total_pnl'].rolling(7, min_periods=1).std().fillna(0)
    daily['vol_bucket'] = pd.cut(daily['pnl_std'], bins=3, labels=['Low', 'Medium', 'High'])

    return daily.dropna(subset=['target'])


def train_model(daily_df):
    feature_cols = ['fg_index', 'sentiment_enc', 'num_trades', 'avg_size',
                    'win_rate', 'ls_ratio', 'total_volume']

    X = daily_df[feature_cols].fillna(0)
    y = daily_df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n--- Model Report (Next-Day Profitability) ---")
    print(classification_report(y_test, y_pred, target_names=['Losing Day', 'Profitable Day']))

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n--- Feature Importances ---")
    print(importances.round(4))

    joblib.dump(model, 'pnl_model.pkl')
    print("\nModel saved to pnl_model.pkl")
    return model, importances


if __name__ == '__main__':
    fg_df = pd.read_csv('fear_greed_index.csv')
    hist_df = pd.read_csv('historical_data.csv')

    fg_df['date'] = pd.to_datetime(fg_df['date']).dt.date
    hist_df['date'] = pd.to_datetime(hist_df['Timestamp IST'], dayfirst=True, errors='coerce').dt.date
    hist_df['Closed PnL'] = pd.to_numeric(hist_df['Closed PnL'], errors='coerce').fillna(0)
    hist_df['Size USD'] = pd.to_numeric(hist_df['Size USD'], errors='coerce').fillna(0)

    daily_df = build_features(hist_df, fg_df)
    daily_df.to_csv('daily_features.csv', index=False)
    print(f"Daily feature table saved: {daily_df.shape[0]} rows")

    model, importances = train_model(daily_df)
