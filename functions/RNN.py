import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

def build_sequences(df, id_col, date_col, num_features, max_len=13, scaler=None):
    df = df.copy()
    
    if scaler is None:
        scaler = StandardScaler()
        df[num_features] = scaler.fit_transform(df[num_features].fillna(0))
    else:
        df[num_features] = scaler.transform(df[num_features].fillna(0))

    df = df.sort_values([id_col, date_col])
    grouped = df.groupby(id_col, sort=False)

    customer_ids = []
    sequences = []
    lengths = []

    for cust_id, g in grouped:
        vals = g[num_features].values.astype('float32')
        seq_len = min(len(vals), max_len)
        vals = vals[-seq_len:]

        padded = np.zeros((max_len, len(num_features)), dtype='float32')
        padded[max_len - seq_len:] = vals   

        sequences.append(padded)
        lengths.append(seq_len)
        customer_ids.append(cust_id)

    return np.stack(sequences), np.array(lengths), np.array(customer_ids), scaler


class CustomerSeqDataset(Dataset):
    def __init__(self, sequences, lengths):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.lengths = torch.tensor(lengths, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.lengths[idx]

def masked_loss(loss_fn, preds, target, lens):
    seq_len = preds.size(1)
    mask = (torch.arange(seq_len, device=preds.device) >= (seq_len - (lens - 1).clamp(0)).unsqueeze(1)).float().unsqueeze(-1)
    return (loss_fn(preds, target) * mask).sum() / mask.sum().clamp(min=1)

def train_autoregressive_rnn(
    sequences, lengths, n_features,
    epochs=50, batch_size=512, lr=1e-3, weight_decay=1e-5,
    val_frac=0.1, patience=5, device='cuda', random_state=42,
):
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    np.random.seed(random_state)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

    val_mask = np.random.RandomState(random_state).rand(len(sequences)) < val_frac
    train_loader = DataLoader(CustomerSeqDataset(sequences[~val_mask], lengths[~val_mask]), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(random_state))
    val_loader = DataLoader(CustomerSeqDataset(sequences[val_mask], lengths[val_mask]), batch_size=batch_size, shuffle=False)

    model = AutoregressiveProfileRNN(feature_dim=n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    loss_fn = nn.HuberLoss(reduction='none')

    def run_epoch(loader, is_train=True):
        model.train(is_train)
        total_loss = 0.0
        with torch.set_grad_enabled(is_train):
            for x, lens in loader:
                x, lens = x.to(device), lens.to(device)
                if is_train: optimizer.zero_grad()
                preds, _ = model(x[:, :-1, :])
                loss = masked_loss(loss_fn, preds, x[:, 1:, :], lens)
                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                total_loss += loss.item()
        return total_loss / len(loader)

    best_val_loss, epochs_no_improve, best_state = float('inf'), 0, None

    for epoch in range(epochs):
        train_loss = run_epoch(train_loader, is_train=True)
        val_loss = run_epoch(val_loader, is_train=False)

        scheduler.step(val_loss)
        print(f'Epoch {epoch+1}/{epochs} — train: {train_loss:.5f}, val: {val_loss:.5f}')

        if val_loss < best_val_loss:
            best_val_loss, epochs_no_improve = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.5f})')
                break

    model.load_state_dict(best_state)
    return model

def extract_embeddings(model, sequences, lengths, device='cuda', batch_size=2048):
    model.eval()
    dataset = CustomerSeqDataset(sequences, lengths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _, hidden = model(x)
            
            flat_hidden = hidden.permute(1, 0, 2).reshape(x.size(0), -1)
            embeddings.append(flat_hidden.cpu().numpy())

    return np.concatenate(embeddings, axis=0)

def forecast_future_profiles(model, sequences, steps_to_forecast=1, device='cuda', batch_size=2048):
    model.eval()
    dataset = CustomerSeqDataset(sequences, np.zeros(len(sequences)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    forecasted_profiles = []
    with torch.no_grad():
        for x, _ in loader:
            current_seq = x.to(device)
            for _ in range(steps_to_forecast):
                preds, _ = model(current_seq)
                next_step = preds[:, -1:, :]
                current_seq = torch.cat([current_seq[:, 1:, :], next_step], dim=1)
            
            forecasted_profiles.append(current_seq[:, -1, :].cpu().numpy())
            
    return np.concatenate(forecasted_profiles, axis=0)