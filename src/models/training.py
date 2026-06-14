from deltalake import DeltaTable

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "src", "models", "lstm_weather.pth")

SEQUENCE_LENGTH = 6
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001

class WeatherLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):

        super(WeatherLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def prepare_data(gold_dir):

    print(f"Loading data from Delta Table at {gold_dir}...")
    if not os.path.exists(gold_dir):
        raise FileNotFoundError(f"Gold directory not found at: {gold_dir}")

    try:
        dt = DeltaTable(gold_dir)
        df = dt.to_pandas()

    except Exception as e:
        raise Exception(f"Failed to read Gold Delta table: {e}")

    print(f"Total records loaded from Delta table: {len(df)}")

    exclude_cols = ['Data Medicao', 'injection_timestamp', 'Window_Reference_Day']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    window_counts = df.groupby('Window_Reference_Day').size()
    valid_windows = window_counts[window_counts == 7].index
    df_filtered = df[df['Window_Reference_Day'].isin(valid_windows)].copy()
    scaler = MinMaxScaler()

    df_filtered[feature_cols] = scaler.fit_transform(df_filtered[feature_cols])
    X, y = [], []

    for ref_day in valid_windows:
        window_data = df_filtered[df_filtered['Window_Reference_Day'] == ref_day][feature_cols].values
        if len(window_data) == 7:
            X.append(window_data[:SEQUENCE_LENGTH])
            y.append(window_data[SEQUENCE_LENGTH])

    return np.array(X), np.array(y), scaler, feature_cols


def train():
    try:
        gold_dir = os.path.join(BASE_DIR, "data", "gold")
        X, y, scaler, feature_cols = prepare_data(gold_dir)
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.222, random_state=42)
        train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=False)

        input_size = len(feature_cols)
        hidden_size = 64
        num_layers = 2
        output_size = len(feature_cols)
        model = WeatherLSTM(input_size, hidden_size, num_layers, output_size)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        print(f"Dataset Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        print(f"Number of features to predict: {output_size}")

        print("Starting training...")
        for epoch in range(EPOCHS):

            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X)
                        val_loss += criterion(outputs, batch_y).item()

                avg_train = train_loss / len(train_loader)
                avg_val = val_loss / len(val_loader)
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

        print("\n" + "="*30)
        print("FINAL EVALUATION (TEST SET)")
        model.eval()
        test_preds = []

        with torch.no_grad():
            for batch_X, _ in test_loader:
                test_preds.append(model(batch_X).numpy())
        y_pred = np.concatenate(test_preds)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"Test MSE (Normalized): {mse:.6f}")
        print(f"Test MAE (Normalized): {mae:.6f}")

        y_test_real = scaler.inverse_transform(y_test)
        y_pred_real = scaler.inverse_transform(y_pred)

        mae_real = mean_absolute_error(y_test_real, y_pred_real)
        print(f"Test MAE (Real Scale - Average across all features): {mae_real:.4f}")

        print("\n" + "-"*30)
        print("SAMPLE PREDICTION VS REAL VALUE")

        for i in range(0, len(y_test_real)):
            sample_idx = i
            print(f"Sample Index: {sample_idx}")
            print(f"{'Feature':<50} | {'Real':>10} | {'Predicted':>10} | {'Error':>10}")
            print("-" * 88)

            for i, col in enumerate(feature_cols):
                real_val = y_test_real[sample_idx, i]
                pred_val = y_pred_real[sample_idx, i]
                error = abs(real_val - pred_val)
                print(f"{col:<50} | {real_val:>10.2f} | {pred_val:>10.2f} | {error:>10.2f}")

            print("-" * 88)

        temp_idx = [i for i, c in enumerate(feature_cols) if 'TEMPERATURA MEDIA' in c.upper()]

        if temp_idx:
            idx = temp_idx[0]
            mae_temp = mean_absolute_error(y_test_real[:, idx], y_pred_real[:, idx])
            print(f"Test MAE for {feature_cols[idx]}: {mae_temp:.4f} °C")

        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'feature_cols': feature_cols,
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'output_size': output_size,
            'y_test_real': y_test_real,
            'y_pred_real': y_pred_real
        }, MODEL_SAVE_PATH)

        print(f"\nModel and metadata saved to {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    train()

