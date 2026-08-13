import numpy as np 
import pandas as pd
from sgt import SGT
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def _build_sequences(df, col, id_col='customer_ID', date_col='S_2'):
    tmp = df[[id_col, date_col, col]].copy()
    tmp[col] = tmp[col].astype('object').where(tmp[col].notna(), 'missing').astype(str)
    tmp = tmp.sort_values([id_col, date_col])

    seqs = (
        tmp.groupby(id_col)[col]
        .apply(list)
        .reset_index()
        .rename(columns={id_col: 'id', col: 'sequence'})
    )
    return seqs


def _sgt_embed_column(df, col, id_col, kappa=5, lengthsensitive=False):
    corpus = _build_sequences(df, col)

    all_symbols = [sym for seq in corpus['sequence'] for sym in seq]
    alphabet = sorted(set(all_symbols))

    sgt = SGT(alphabets=alphabet, kappa=kappa,
              lengthsensitive=lengthsensitive, flatten=True)
    emb = sgt.fit_transform(corpus)

    emb = emb.set_index('id')
    emb.columns = [f'{col}_sgt_{a}_{b}' for (a, b) in emb.columns]
    emb = emb.reset_index().rename(columns={'id': id_col})
    return emb


def build_sgt_features(df, cat_cols=CAT_COLS, id_col, kappa=5, lengthsensitive=False, verbose=True):
    feature_frames = []
    for col in cat_cols:
        if verbose:
            print(f'Fitting SGT on {col} ...')
        emb = _sgt_embed_column(df, col, kappa=kappa,
                                 lengthsensitive=lengthsensitive)
        feature_frames.append(emb)

    merged = feature_frames[0]
    for emb in feature_frames[1:]:
        merged = merged.merge(emb, on=id_col, how='outer')

    merged = merged.fillna(0.0)
    return merged


def add_anomaly_scores(feat_df, id_col, contamination=0.02, random_state=42):
    """
    Adds sgt_anomaly_flag (-1 anomalous / 1 normal) and a continuous
    sgt_anomaly_score (higher = more anomalous).
    """
    feature_cols = [c for c in feat_df.columns if c != id_col]
    X = feat_df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(n_estimators=300, contamination=contamination,
                           random_state=random_state, n_jobs=-1)
    iso.fit(X_scaled)

    out = feat_df.copy()
    out['sgt_anomaly_flag'] = iso.predict(X_scaled)
    out['sgt_anomaly_score'] = -iso.score_samples(X_scaled)
    return out

def add_centroid_distance(feat_df, id_col):
    feature_cols = [c for c in feat_df.columns
                     if c not in (id_col, 'sgt_anomaly_flag', 'sgt_anomaly_score')]
    X = feat_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    centroid = X_scaled.mean(axis=0)
    dist = np.linalg.norm(X_scaled - centroid, axis=1)

    out = feat_df.copy()
    out['sgt_centroid_distance'] = dist
    return out

def filter_sgt_features(sgt_feats, id_col='customer_ID', target=None, var_threshold=1e-4, max_features=50):
    """
    Reduces a wide SGT embedding table down to a small, useful set of
    columns before it ever touches the main dataframe. Unaffected by
    main_df's granularity — sgt_feats stays one row per customer_ID.
    """
    always_keep = [c for c in
                   ['sgt_anomaly_flag', 'sgt_anomaly_score', 'sgt_centroid_distance']
                   if c in sgt_feats.columns]

    candidate_cols = [c for c in sgt_feats.columns
                       if c not in always_keep + [id_col]]

    variances = sgt_feats[candidate_cols].var()
    candidate_cols = variances[variances > var_threshold].index.tolist()
    print(f'Dropped {sgt_feats.shape[1] - len(always_keep) - 1 - len(candidate_cols)} '
          f'near-constant columns (variance <= {var_threshold})')

    if target is not None:
        aligned_target = target.reindex(sgt_feats[id_col]).values if hasattr(target, 'reindex') else target
        corr = sgt_feats[candidate_cols].corrwith(pd.Series(aligned_target)).abs()
        keep_cols = corr.sort_values(ascending=False).head(max_features).index.tolist()
        print(f'Kept top {len(keep_cols)} SGT columns by |correlation with target|')
    else:
        keep_cols = sgt_feats[candidate_cols].var().sort_values(ascending=False) \
                        .head(max_features).index.tolist()
        print(f'Kept top {len(keep_cols)} SGT columns by variance')

    final_cols = [id_col] + always_keep + keep_cols
    out = sgt_feats[final_cols].copy()

    float_cols = out.select_dtypes(include=['float64']).columns
    out[float_cols] = out[float_cols].astype('float32')

    print(f'Final SGT feature block: {out.shape[1]} columns '
          f'(down from {sgt_feats.shape[1]})')
    return out

def process_and_feature_engineer(df, id_col, cat_features, front_cols, date_col):
    all_cols = [c for c in df.columns if c not in front_cols]
    num_features = [c for c in all_cols if c not in cat_features]

    df = df.sort_values([id_col, date_col])
    df[date_col] = pd.to_datetime(df[date_col])

    agg_dict = {c: ['mean', 'std', 'min', 'max', 'last', 'first'] for c in num_features}
    agg_dict.update({c: ['last', 'nunique'] for c in cat_features})
    agg_dict[date_col] = ['first', 'last', 'count']

    monthly_df = df.groupby(id_col, sort=False).agg(agg_dict)
    monthly_df.columns = ['_'.join(col).strip('_') for col in monthly_df.columns]
    monthly_df = monthly_df.reset_index()
    monthly_df = monthly_df.copy()
    
    derived = {}
    derived[f'{date_col}_n_statements'] = monthly_df[f'{date_col}_count']
    derived['tenure_days'] = (
        monthly_df[f'{date_col}_last'] - monthly_df[f'{date_col}_first']
    ).dt.days

    for c in num_features:
        mean_val = monthly_df[f'{c}_mean']
        last_val = monthly_df[f'{c}_last']
        first_val = monthly_df[f'{c}_first']
        derived[f'{c}_last_minus_mean'] = last_val - mean_val
        derived[f'{c}_last_div_mean'] = last_val / mean_val.replace(0, np.nan)
        derived[f'{c}_last_minus_first'] = last_val - first_val

    derived['n_missing_num'] = monthly_df[[f'{c}_mean' for c in num_features]].isna().sum(axis=1)
    derived['pct_missing_num'] = derived['n_missing_num'] / max(len(num_features), 1)

    monthly_df = pd.concat([monthly_df, pd.DataFrame(derived)], axis=1)
    monthly_df = monthly_df.copy() 

    front_present = [c for c in front_cols if c in monthly_df.columns]
    other_cols = [c for c in monthly_df.columns if c not in front_present]
    final_df = monthly_df[front_present + other_cols]

    return final_df