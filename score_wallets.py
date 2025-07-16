import pandas as pd
import numpy as np
import argparse
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file):
    with open(file, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df

def engineer_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    grouped = df.groupby('userWallet').agg({
        'action': ['count'],
        'timestamp': [lambda x: (x.max() - x.min()).days]
    })
    grouped.columns = ['txn_count', 'days_active']

    action_counts = df.pivot_table(index='userWallet', columns='action', aggfunc='size', fill_value=0)
    features = grouped.join(action_counts)
    features.fillna(0, inplace=True)
    return features

def save_model(model, scaler):
    print("Model and scaler are being saved...")
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Model and scaler are being saved...")

def train_model(features):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)
    model = IsolationForest(random_state=42)
    model.fit(scaled)
    print("Model trained successfully, now saving...")
    save_model(model, scaler)
    return scaler, model

def score_wallets(features, scaler, model):
    scaled = scaler.transform(features)
    anomaly_scores = -model.decision_function(scaled)
    credit_scores = MinMaxScaler((0, 1000)).fit_transform(anomaly_scores.reshape(-1, 1)).flatten()
    results = pd.DataFrame({
        'wallet': features.index.astype(str),
        'score': credit_scores.astype(int)
    })
    return results

def plot_distribution(scores_df):
    sns.histplot(scores_df['score'], bins=10, kde=False)
    plt.title('Wallet Credit Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.savefig('score_distribution.png')

def main(args):
    df = load_data(args.input)
    features = engineer_features(df)
    scaler, model = train_model(features)
    scores = score_wallets(features, scaler, model)
    scores.to_csv(args.output, index=False)
    plot_distribution(scores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    main(args)
