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
    prefixes = ["D_", "S_", "P_", "B_", "R_"]

    df = df.sort_values([id_col, date_col])
    df[date_col] = pd.to_datetime(df[date_col])
    df['month'] = train_df[date_col].dt.strftime("%Y-%m")

    stmt_counts = (
        df.groupby(id_col, sort=False)[date_col]
        .count()
        .rename('n_statements')
        .reset_index()
    )

    month_summary = (
        df.groupby(id_col, sort=False)['month']
        .agg(last_stmt_month='last', first_stmt_month='first')
        .reset_index()
    )

    agg_dict = {c: "mean" for c in num_features}
    agg_dict.update({c: "last" for c in cat_features})
    agg_dict[date_col] = "last"
    agg_dict['month'] = "last"

    monthly_df = (
        df.groupby(id_col, as_index=False, sort=False)
        .agg(agg_dict)
        .sort_values(id_col)
        .reset_index(drop=True)
    )
    monthly_df = monthly_df[
        front_cols + [c for c in monthly_df.columns if c not in front_cols]
    ]

    grouped = monthly_df.groupby(id_col, sort=False)
    new_cols = {}
    new_cols["n_missing_num"] = monthly_df[num_features].isna().sum(axis=1)
    new_cols["pct_missing_num"] = new_cols["n_missing_num"] / max(len(num_features), 1)
    prev_s2 = grouped[date_col].shift(1)
    new_cols["days_since_prev_stmt"] = (pd.to_datetime(monthly_df[date_col]) - pd.to_datetime(prev_s2)).dt.days
    new_cols["stmt_index"] = grouped.cumcount() + 1
    new_cols["n_statements_total"] = grouped[date_col].transform('count')

    cat_mean_cols = {}
    for p in prefixes:
        p_cols = [c for c in num_features if c.startswith(p)]
        if p_cols:
            col_name = f"{p}row_mean"
            new_cols[col_name] = monthly_df[p_cols].mean(axis=1)
            cat_mean_cols[p] = col_name
    if "B_" in cat_mean_cols and "S_" in cat_mean_cols:
        new_cols["B_over_S"] = new_cols[cat_mean_cols["B_"]] / new_cols[cat_mean_cols["S_"]].replace(0, np.nan)
    if "P_" in cat_mean_cols and "B_" in cat_mean_cols:
        new_cols["P_over_B"] = new_cols[cat_mean_cols["P_"]] / new_cols[cat_mean_cols["B_"]].replace(0, np.nan)
    if "R_" in cat_mean_cols and "D_" in cat_mean_cols:
        new_cols["R_times_D"] = new_cols[cat_mean_cols["R_"]] * new_cols[cat_mean_cols["D_"]]

    for c in cat_features:
        changed = (monthly_df[c] != grouped[c].shift(1)).astype(int)
        is_first = new_cols["stmt_index"] == 1
        changed[is_first] = 0
        new_cols[f"{c}_changed"] = changed

    monthly_df = pd.concat([monthly_df, pd.DataFrame(new_cols)], axis=1)
    monthly_df = monthly_df.copy()

    summary_cols = {}
    for c in num_features:
        cust_group = monthly_df.groupby(id_col, sort=False)[c]
        mean_val = cust_group.transform('mean')
        last_val = cust_group.transform('last')
        first_val = cust_group.transform('first')
        summary_cols[f"{c}_mean"] = mean_val
        summary_cols[f"{c}_std"] = cust_group.transform('std')
        summary_cols[f"{c}_min"] = cust_group.transform('min')
        summary_cols[f"{c}_max"] = cust_group.transform('max')
        summary_cols[f"{c}_last"] = last_val
        summary_cols[f"{c}_last_minus_mean"] = last_val - mean_val
        summary_cols[f"{c}_last_div_mean"] = last_val / mean_val.replace(0, np.nan)
        summary_cols[f"{c}_last_minus_first"] = last_val - first_val

    for c in cat_features:
        cust_group = monthly_df.groupby(id_col, sort=False)[c]
        summary_cols[f"{c}_last_cat"] = cust_group.transform('last')
        summary_cols[f"{c}_nunique_cat"] = cust_group.transform('nunique')

    flag_cols = [k for k in new_cols if k.endswith("_changed")]
    skip_cols = {"stmt_index", "n_statements_total"} | set(flag_cols)
    eng_summary_targets = [k for k in new_cols if k not in skip_cols]

    for c in eng_summary_targets:
        cust_group = monthly_df.groupby(id_col, sort=False)[c]
        mean_val = cust_group.transform('mean')
        last_val = cust_group.transform('last')
        summary_cols[f"{c}_mean"] = mean_val
        summary_cols[f"{c}_std"] = cust_group.transform('std')
        summary_cols[f"{c}_max"] = cust_group.transform('max')
        summary_cols[f"{c}_last"] = last_val

    for c in flag_cols:
        cust_group = monthly_df.groupby(id_col, sort=False)[c]
        summary_cols[f"{c}_count"] = cust_group.transform('sum')
        summary_cols[f"{c}_last"] = cust_group.transform('last')

    summary_df = pd.concat(
        [monthly_df[[id_col]], pd.DataFrame(summary_cols)],
        axis=1
    )
    summary_df = summary_df.copy()
    final_df = (
        summary_df
        .drop_duplicates(subset=id_col, keep='last')
        .reset_index(drop=True)
        .copy()
    )

    final_df = final_df.merge(stmt_counts, on=id_col, how='left')
    final_df = final_df.merge(month_summary, on=id_col, how='left').copy()

    return final_df
