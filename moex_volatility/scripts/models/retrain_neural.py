import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

def qlike(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, None)
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)

# Load data
train = pd.read_parquet('data/prepared/train.parquet')
val = pd.read_parquet('data/prepared/val.parquet')
test = pd.read_parquet('data/prepared/test.parquet')

with open('data/prepared/config.json') as f:
    config = json.load(f)
feature_cols = config['feature_cols']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

SEQ_LEN = 22

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, rnn_type='GRU'):
        super().__init__()
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.rnn_type = rnn_type

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)

def create_sequences(df, feature_cols, target_col, seq_len):
    sequences, targets, meta = [], [], []
    for ticker, group in df.groupby('ticker'):
        group = group.sort_values('date').reset_index(drop=True)
        feats = group[feature_cols].values
        tgt = group[target_col].values
        dates = group['date'].values
        for i in range(seq_len, len(group)):
            seq = feats[i-seq_len:i]
            if not np.isnan(seq).any() and not np.isnan(tgt[i]):
                sequences.append(seq)
                targets.append(tgt[i])
                meta.append({'date': dates[i], 'ticker': ticker, 'rv_actual': tgt[i]})
    return np.array(sequences), np.array(targets), meta

for h in [1, 5, 22]:
    target = f'rv_target_h{h}'
    print(f'\n========== H={h} ==========')

    # Проверь что target существует
    for name, df in [('train', train), ('val', val), ('test', test)]:
        if target not in df.columns:
            print(f'ERROR: {target} not in {name} columns')
            continue

    # Create sequences
    X_train_seq, y_train_raw, _ = create_sequences(train, feature_cols, target, SEQ_LEN)
    X_val_seq, y_val_raw, _ = create_sequences(val, feature_cols, target, SEQ_LEN)
    X_test_seq, y_test_raw, meta_test = create_sequences(test, feature_cols, target, SEQ_LEN)

    print(f'Train: {X_train_seq.shape}, Val: {X_val_seq.shape}, Test: {X_test_seq.shape}')

    # Scale features (fit on train only)
    n_feat = X_train_seq.shape[2]
    scaler_X = StandardScaler()
    X_train_flat = X_train_seq.reshape(-1, n_feat)
    scaler_X.fit(X_train_flat)

    # Replace inf/nan before scaling
    X_train_seq = np.nan_to_num(X_train_seq, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_seq = np.nan_to_num(X_val_seq, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_seq = np.nan_to_num(X_test_seq, nan=0.0, posinf=0.0, neginf=0.0)

    X_train_scaled = scaler_X.transform(X_train_seq.reshape(-1, n_feat)).reshape(X_train_seq.shape)
    X_val_scaled = scaler_X.transform(X_val_seq.reshape(-1, n_feat)).reshape(X_val_seq.shape)
    X_test_scaled = scaler_X.transform(X_test_seq.reshape(-1, n_feat)).reshape(X_test_seq.shape)

    # NaN safety after scaling
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Target: log transform (ВАЖНО — обратная трансформация через exp)
    y_train_log = np.log(y_train_raw + 1e-10)
    y_val_log = np.log(y_val_raw + 1e-10)

    # Scale target too
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_log.reshape(-1,1)).ravel()
    y_val_scaled = scaler_y.transform(y_val_log.reshape(-1,1)).ravel()

    # DataLoaders
    train_ds = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled))
    val_ds = TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val_scaled))
    test_ds = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(np.zeros(len(X_test_scaled))))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)
    test_loader = DataLoader(test_ds, batch_size=256)

    for rnn_type in ['GRU', 'LSTM']:
        print(f'\n--- {rnn_type} H={h} ---')

        model = RNNModel(n_feat, hidden_size=64, num_layers=2, dropout=0.2, rnn_type=rnn_type).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(100):
            # Train
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            # Val
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    pred = model(X_batch)
                    val_loss += criterion(pred, y_batch).item()

            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= 15:
                print(f'  Early stop epoch {epoch}')
                break

            if epoch % 10 == 0:
                print(f'  Epoch {epoch}: train_loss={train_loss/len(train_loader):.6f}, val_loss={val_loss:.6f}')

        # Load best
        model.load_state_dict(best_state)
        model.eval()

        # Predict on test
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                pred = model(X_batch)
                all_preds.append(pred.cpu().numpy())

        preds_scaled = np.concatenate(all_preds)

        # ОБРАТНАЯ ТРАНСФОРМАЦИЯ (ключевой момент!)
        preds_log = scaler_y.inverse_transform(preds_scaled.reshape(-1,1)).ravel()
        preds_rv = np.exp(preds_log)  # из log-space обратно
        preds_rv = np.clip(preds_rv, 1e-10, None)

        q = qlike(y_test_raw, preds_rv)
        print(f'  {rnn_type} H={h} QLIKE = {q:.4f}')
        print(f'  pred stats: mean={preds_rv.mean():.6f}, min={preds_rv.min():.6f}, max={preds_rv.max():.6f}')
        print(f'  actual stats: mean={y_test_raw.mean():.6f}')

        # Save model
        model_dir = Path(f'models/{rnn_type.lower()}/h{h}')
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, model_dir / 'model.pt')

        # Save predictions (УНИФИЦИРОВАННЫЙ формат)
        pred_df = pd.DataFrame(meta_test)
        pred_df['rv_pred'] = preds_rv
        pred_df.to_parquet(f'data/predictions/test_2019/{rnn_type.lower()}_h{h}.parquet', index=False)

print('\n=== DONE ===')
