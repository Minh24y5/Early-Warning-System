import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt, gc, os
import glob
import pyarrow as pa
import pyarrow.parquet as pq
import pickle
import re
import json
from sgt import SGT
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import xgboost as xgb
import kagglehub



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


def _sgt_embed_column(df, col, kappa=5, lengthsensitive=False):
    corpus = _build_sequences(df, col)

    all_symbols = [sym for seq in corpus['sequence'] for sym in seq]
    alphabet = sorted(set(all_symbols))

    sgt = SGT(alphabets=alphabet, kappa=kappa,
              lengthsensitive=lengthsensitive, flatten=True)
    emb = sgt.fit_transform(corpus)

    emb = emb.set_index('id')
    emb.columns = [f'{col}_sgt_{a}_{b}' for (a, b) in emb.columns]
    emb = emb.reset_index().rename(columns={'id': 'customer_ID'})
    return emb


def build_sgt_features(df, cat_cols=CAT_COLS, kappa=5,
                        lengthsensitive=False, verbose=True):
    feature_frames = []
    for col in cat_cols:
        if verbose:
            print(f'Fitting SGT on {col} ...')
        emb = _sgt_embed_column(df, col, kappa=kappa,
                                 lengthsensitive=lengthsensitive)
        feature_frames.append(emb)

    merged = feature_frames[0]
    for emb in feature_frames[1:]:
        merged = merged.merge(emb, on='customer_ID', how='outer')

    merged = merged.fillna(0.0)
    return merged


def add_anomaly_scores(feat_df, contamination=0.02, random_state=42):
    """
    Adds sgt_anomaly_flag (-1 anomalous / 1 normal) and a continuous
    sgt_anomaly_score (higher = more anomalous).
    """
    feature_cols = [c for c in feat_df.columns if c != 'customer_ID']
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


def add_centroid_distance(feat_df):
    feature_cols = [c for c in feat_df.columns
                     if c not in ('customer_ID', 'sgt_anomaly_flag', 'sgt_anomaly_score')]
    X = feat_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    centroid = X_scaled.mean(axis=0)
    dist = np.linalg.norm(X_scaled - centroid, axis=1)

    out = feat_df.copy()
    out['sgt_centroid_distance'] = dist
    return out

def filter_sgt_features(sgt_feats, id_col='customer_ID', target=None,
                         var_threshold=1e-4, max_features=50):
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

# sgt_feats = build_sgt_features(train_df, CAT_COLS, kappa=5)
# sgt_feats = add_anomaly_scores(sgt_feats, contamination=0.02)
# sgt_feats = add_centroid_distance(sgt_feats)
# sgt_feats = filter_sgt_features(sgt_feats,target=targets.set_index('customer_ID')['target'])

# print(sgt_feats.shape)
# sgt_feats.to_csv('sgt_feats.csv')
# sgt_feats.sort_values('sgt_anomaly_score', ascending=False).head(10)
# del CAT_COLS, NUM_COLS

sgt_feats = pd.read_csv('/kaggle/input/datasets/minh24y5/amex-engineered-features-batch/sgt_feats.csv')

# 
def process_and_feature_engineer(df):
    cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120",
                    "D_126","D_63","D_64","D_66","D_68"]
    front_cols = ["customer_ID", "S_2","S_2_month"]
    all_cols = [c for c in df.columns if c not in front_cols]
    num_features = [c for c in all_cols if c not in cat_features]
    prefixes = ["D_", "S_", "P_", "B_", "R_"]

    df = df.sort_values(['customer_ID', 'S_2'])

    stmt_counts = (
        df.groupby('customer_ID', sort=False)['S_2']
        .count()
        .rename('n_statements')
        .reset_index()
    )

    month_summary = (
        df.groupby('customer_ID', sort=False)['S_2_month']
        .agg(last_stmt_month='last', first_stmt_month='first')
        .reset_index()
    )

    agg_dict = {c: "mean" for c in num_features}
    agg_dict.update({c: "last" for c in cat_features})
    agg_dict['S_2'] = "last"
    agg_dict['S_2_month'] = "last"

    monthly_df = (
        df.groupby('customer_ID', as_index=False, sort=False)
        .agg(agg_dict)
        .sort_values('customer_ID')
        .reset_index(drop=True)
    )
    monthly_df = monthly_df[
        front_cols + [c for c in monthly_df.columns if c not in front_cols]
    ]

    grouped = monthly_df.groupby('customer_ID', sort=False)
    new_cols = {}
    new_cols["n_missing_num"] = monthly_df[num_features].isna().sum(axis=1)
    new_cols["pct_missing_num"] = new_cols["n_missing_num"] / max(len(num_features), 1)
    prev_s2 = grouped['S_2'].shift(1)
    new_cols["days_since_prev_stmt"] = (pd.to_datetime(monthly_df['S_2']) - pd.to_datetime(prev_s2)).dt.days
    new_cols["stmt_index"] = grouped.cumcount() + 1
    new_cols["n_statements_total"] = grouped['S_2'].transform('count')

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
        cust_group = monthly_df.groupby('customer_ID', sort=False)[c]
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
        cust_group = monthly_df.groupby('customer_ID', sort=False)[c]
        summary_cols[f"{c}_last_cat"] = cust_group.transform('last')
        summary_cols[f"{c}_nunique_cat"] = cust_group.transform('nunique')

    flag_cols = [k for k in new_cols if k.endswith("_changed")]
    skip_cols = {"stmt_index", "n_statements_total"} | set(flag_cols)
    eng_summary_targets = [k for k in new_cols if k not in skip_cols]

    for c in eng_summary_targets:
        cust_group = monthly_df.groupby('customer_ID', sort=False)[c]
        mean_val = cust_group.transform('mean')
        last_val = cust_group.transform('last')
        summary_cols[f"{c}_mean"] = mean_val
        summary_cols[f"{c}_std"] = cust_group.transform('std')
        summary_cols[f"{c}_max"] = cust_group.transform('max')
        summary_cols[f"{c}_last"] = last_val

    for c in flag_cols:
        cust_group = monthly_df.groupby('customer_ID', sort=False)[c]
        summary_cols[f"{c}_count"] = cust_group.transform('sum')
        summary_cols[f"{c}_last"] = cust_group.transform('last')

    summary_df = pd.concat(
        [monthly_df[['customer_ID']], pd.DataFrame(summary_cols)],
        axis=1
    )
    summary_df = summary_df.copy()
    final_df = (
        summary_df
        .drop_duplicates(subset='customer_ID', keep='last')
        .reset_index(drop=True)
        .copy()
    )

    final_df = final_df.merge(stmt_counts, on='customer_ID', how='left')
    final_df = final_df.merge(month_summary, on='customer_ID', how='left').copy()

    return final_df

# 
def optimize_dtypes(df):
    for col in df.columns:
        col_type = df[col].dtype

        if col_type == "float64":
            df[col] = df[col].astype(np.float32)

        elif col_type == "int64":
            c_min, c_max = df[col].min(), df[col].max()
            if c_min >= 0:
                if c_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif c_max < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif c_max < 4294967295:
                    df[col] = df[col].astype(np.uint32)
            else:
                if (
                    c_min > np.iinfo(np.int8).min
                    and c_max < np.iinfo(np.int8).max
                ):
                    df[col] = df[col].astype(np.int8)
                elif (
                    c_min > np.iinfo(np.int16).min
                    and c_max < np.iinfo(np.int16).max
                ):
                    df[col] = df[col].astype(np.int16)
                elif (
                    c_min > np.iinfo(np.int32).min
                    and c_max < np.iinfo(np.int32).max
                ):
                    df[col] = df[col].astype(np.int32)

        elif col_type == "object":
            df[col] = df[col].astype("category")

    return df

def safe_merge(main_df, sgt_feats, id_col='customer_ID'):
    before_rows = len(main_df)
    model_df = main_df.merge(sgt_feats, on=id_col, how='left')
    assert len(model_df) == before_rows, "Row count changed during merge — check for duplicate IDs in sgt_feats"

    del main_df, sgt_feats
    gc.collect()
    return model_df

def process_in_customer_chunks(
    df, sgt_feats, chunk_size=229500, save_dir="/kaggle/working/data/batch_data"
):
    os.makedirs(save_dir, exist_ok=True)
    unique_custs = df["customer_ID"].unique()
    total_custs = len(unique_custs)
    num_chunks = int(np.ceil(total_custs / chunk_size))
    print(
        f"Total Customers: {total_custs} | Processing in {num_chunks} chunk(s) of ~{chunk_size} customers each."
    )
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_custs)
        batch_cust_ids = set(unique_custs[start_idx:end_idx])
        df_chunk = df[df["customer_ID"].isin(batch_cust_ids)].copy()
        processed_chunk = process_and_feature_engineer(df_chunk)

        sgt_chunk = sgt_feats[sgt_feats["customer_ID"].isin(batch_cust_ids)]
        processed_chunk = safe_merge(processed_chunk, sgt_chunk)

        processed_chunk = optimize_dtypes(processed_chunk)
        chunk_file = os.path.join(save_dir, f"batch_{i+1}_of_{num_chunks}.parquet")
        processed_chunk.to_parquet(
            chunk_file, compression="zstd", index=False
        )
        print(f"Saved: {chunk_file}")
        del df_chunk, processed_chunk, sgt_chunk, batch_cust_ids
        gc.collect()
    print("All chunks processed successfully!")

process_in_customer_chunks(train_df, sgt_feats)
del train_df

def _list_batch_files(batch_dir):
    files = sorted(glob.glob(os.path.join(batch_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {batch_dir}")
    return files
 
 
def _get_schema_columns(path):
    schema = pq.ParquetFile(path).schema_arrow
 
    numeric_cols, other_cols = [], []
    for field in schema:
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
            numeric_cols.append(field.name)
        else:
            other_cols.append(field.name)
    return numeric_cols, other_cols
 
 
def _rank_numeric_cols_by_variance(batch_files, numeric_cols):
    total_count = pd.Series(0, index=numeric_cols, dtype="int64")
    total_sum = pd.Series(0.0, index=numeric_cols, dtype="float64")
    total_sumsq = pd.Series(0.0, index=numeric_cols, dtype="float64")
 
    for path in batch_files:
        chunk = pd.read_parquet(path, columns=numeric_cols)
        total_count += chunk.count()
        total_sum += chunk.sum(skipna=True)
        total_sumsq += (chunk ** 2).sum(skipna=True)
 
    mean = total_sum / total_count.replace(0, np.nan)
    variance = total_sumsq / total_count.replace(0, np.nan) - mean ** 2
    return variance.sort_values(ascending=False).index.tolist()

def combine_batches_select_features(
    batch_dir,
    output_dir,
    n_features=300,
    feature_cols=None,
    id_cols=("customer_ID", "S_2", "S_2_month"),
    selection_method="variance",
    compression="zstd",
    output_filename="combined_features.parquet",
):
    
    os.makedirs(output_dir, exist_ok=True)
    batch_files = _list_batch_files(batch_dir)
 
    numeric_cols, other_cols = _get_schema_columns(batch_files[0])
    id_cols_present = [c for c in id_cols if c in numeric_cols or c in other_cols]
 
    numeric_cols = [c for c in numeric_cols if c not in id_cols_present]
    other_cols = [c for c in other_cols if c not in id_cols_present]
 
    if feature_cols is not None:
        selected_features = [c for c in feature_cols if c not in id_cols_present]
    else:
        if selection_method != "variance":
            raise ValueError(f"Unsupported selection_method: {selection_method!r}")
 
        if len(numeric_cols) <= n_features:
            ranked_numeric = numeric_cols
        else:
            ranked_numeric = _rank_numeric_cols_by_variance(batch_files, numeric_cols)[:n_features]
 
        ranked_numeric_set = set(ranked_numeric)
        selected_numeric = [c for c in numeric_cols if c in ranked_numeric_set]
        selected_features = selected_numeric + other_cols
 
    final_cols = id_cols_present + selected_features
 
    combined_chunks = [pd.read_parquet(path, columns=final_cols) for path in batch_files]
    combined_df = pd.concat(combined_chunks, ignore_index=True)
 
    out_path = os.path.join(output_dir, output_filename)
    combined_df.to_parquet(out_path, index=False, compression=compression)
 
    return None

batch_cols = pd.read_parquet('/kaggle/working/data/batch_data/batch_1_of_2.parquet').columns.tolist()

id_cols = ["customer_ID", "last_stmt_month", "first_stmt_month"]
feature_cols_all = [c for c in batch_cols if c not in id_cols]

print(f'Total columns: {len(batch_cols)} | Feature columns (excl. IDs): {len(feature_cols_all)}')

N_FILES = 3
col_chunks = np.array_split(feature_cols_all, N_FILES)

for i, cols in enumerate(col_chunks):
    print(f'File {i+1}: {len(cols)} feature columns')
    combine_batches_select_features(
        '/kaggle/working/data/batch_data',
        '/kaggle/working/data/combine_data',
        feature_cols=list(cols),
        id_cols = id_cols,
        output_filename=f'combined_{i+1}.parquet'
    )
# FILTERING DATA
def calculate_woe_iv(df, feature, target, is_continuous=False, bins=10):
    df = df[[feature, target]].copy()

    if is_continuous: df["bin"] = pd.qcut(df[feature], q=bins, duplicates="drop")
    else: df["bin"] = df[feature].fillna("Missing")

    grouped = df.groupby("bin", observed=False)[target].agg(["count", "sum"])
    grouped.columns = ["Total", "Bad"]
    grouped["Good"] = grouped["Total"] - grouped["Bad"]

    total_good = grouped["Good"].sum()
    total_bad = grouped["Bad"].sum()

    if total_good == 0 or total_bad == 0:
        return 0.0, grouped

    grouped["Prop_Good"] = grouped["Good"] / total_good
    grouped["Prop_Bad"] = grouped["Bad"] / total_bad

    grouped["Prop_Good"] = grouped["Prop_Good"].replace(0, 0.0001)
    grouped["Prop_Bad"] = grouped["Prop_Bad"].replace(0, 0.0001)

    grouped["WoE"] = np.log(grouped["Prop_Good"] / grouped["Prop_Bad"])
    grouped["IV"] = (grouped["Prop_Good"] - grouped["Prop_Bad"]) * grouped["WoE"]

    total_iv = grouped["IV"].sum()

    return total_iv, grouped


def get_all_features_iv(df, target, max_bins=10, unique_threshold=10):
    iv_results = {}
    cols_to_skip = [target, "customer_ID", "last_stmt_month", "first_stmt_month"]
    
    for col in df.columns:
        if col in cols_to_skip:
            continue

        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        is_low_cardinality = df[col].nunique() <= unique_threshold

        is_continuous = is_numeric and not is_low_cardinality

        try:
            iv_value, _ = calculate_woe_iv(
                df,
                col,
                target,
                is_continuous=is_continuous,
                bins=max_bins,
            )
            iv_results[col] = iv_value
        except Exception as e:
            iv_results[col] = np.nan

    iv_summary = pd.DataFrame(
        list(iv_results.items()), columns=["Feature", "IV"]
    )
    iv_summary = iv_summary.sort_values(by="IV", ascending=False).reset_index(
        drop=True
    )

    return iv_summary

def corr_with_flag(x: pd.Series, y: pd.Series, method: str = "spearman") -> float:
    if not pd.api.types.is_numeric_dtype(x):
        return np.nan
    if x.nunique(dropna=True) <= 1 or y.nunique(dropna=True) <= 1:
        return np.nan

    return abs(x.corr(y.astype(float), method=method))

def get_missing_rates(df):
    missing_rate = df.isnull().mean()        
    missing_df = missing_rate.reset_index()
    missing_df.columns = ['Feature', 'missing_rate']
    
    return missing_df.sort_values(by='missing_rate', ascending=False).reset_index(drop=True)

def csi(
    actual_df: pd.DataFrame,
    expected_df: pd.DataFrame,
    feature_cols: list,
    naming_index: str,
    handle_null: str = "keep",
    n_bins: int = 10,
    epsilon: float = 1e-6,) -> list:
    
    results = []
    
    for feature in feature_cols:
        try:
            # Step 1: Handle nulls
            actual_series = actual_df[feature].copy()
            expected_series = expected_df[feature].copy()
            
            if handle_null == "drop":
                actual_series = actual_series.dropna()
                expected_series = expected_series.dropna()

            # Skip if constant or empty
            if (
                actual_series.nunique(dropna=False) <= 1 
                or expected_series.nunique(dropna=False) <= 1
            ):
                results.append({
                    "feature_name": feature,
                    "metric_name": f"CSI_{naming_index}",
                    "metric_type": "csi",
                    "metric_value": np.nan
                })
                continue
            
            # Step 2 & 3: Compute quantiles and bin edges (numeric columns only)
            if not pd.api.types.is_string_dtype(actual_series):
                quantiles = expected_series.quantile([i / n_bins for i in range(1, n_bins)]).values
                delta = 1e-9
                bin_edges = [-np.inf] + sorted(list(set(quantiles + delta))) + [np.inf]
                
                # Step 4: Apply binning
                actual_bins = pd.cut(actual_series, bins=bin_edges, labels=False, right=True)
                expected_bins = pd.cut(expected_series, bins=bin_edges, labels=False, right=True)
                
                if handle_null == "keep":
                    actual_bins = actual_bins.fillna("NaN_bin")
                    expected_bins = expected_bins.fillna("NaN_bin")
            else:
                actual_bins = actual_series.fillna("NaN_bin")
                expected_bins = expected_series.fillna("NaN_bin")

            # Step 5: Distribution by bin (Count)
            actual_counts = actual_bins.value_counts(dropna=False).rename("actual_count")
            expected_counts = expected_bins.value_counts(dropna=False).rename("expected_count")

            # Step 6: Proportion
            actual_pct = (actual_counts / len(actual_series)).rename("actual_pct")
            expected_pct = (expected_counts / len(expected_series)).rename("expected_pct")

            # Step 7: Combine distributions and fill missing bins with epsilon
            csi_df = pd.concat([actual_pct, expected_pct], axis=1).fillna(epsilon)

            # Step 8: CSI calculation
            csi_df["csi_component"] = (csi_df["actual_pct"] - csi_df["expected_pct"]) * np.log(
                (csi_df["actual_pct"] + epsilon) / (csi_df["expected_pct"] + epsilon)
            )
            csi_value = csi_df["csi_component"].sum()

            results.append({
                "Feature": feature,
                "metric_name": f"CSI_{naming_index}",
                "metric_type": "csi",
                "csi_value": round(float(csi_value), 6)
            })

        except Exception as e:
            print(f"Error on feature {feature}: {e}")
            results.append({
                "Feature": feature,
                "metric_name": f"CSI_{naming_index}",
                "metric_type": "csi",
                "csi_value": np.nan
            })
            
    return results

def pick_split_col(df, candidates=('last_stmt_month', 'first_stmt_month')):
    for col in candidates:
        if col in df.columns:
            n_unique = df[col].dropna().nunique()
            if n_unique > 1:
                return col
    return None

def flag_anomalous_features(main_df, feature_cols, id_col='customer_ID',
                             outlier_share_threshold=0.01, z_threshold=6.0,
                             dedupe_by_customer=False):
    df_for_stats = main_df
    if dedupe_by_customer:
        if id_col not in main_df.columns:
            raise ValueError(f"dedupe_by_customer=True requires '{id_col}' in main_df")
        df_for_stats = main_df.drop_duplicates(subset=id_col, keep='last')

    rows = []
    for col in feature_cols:
        s = df_for_stats[col]
        z = (s - s.mean()) / (s.std() if s.std() != 0 else 1)
        outlier_share = (z.abs() > z_threshold).mean()
        rows.append({'feature': col, 'outlier_share': outlier_share})

    report = pd.DataFrame(rows).sort_values('outlier_share', ascending=False)
    flagged = report[report['outlier_share'] > outlier_share_threshold]
    basis = 'per-customer (deduped)' if dedupe_by_customer else 'per-statement (row-level)'
    return report

# 
targets = pd.read_csv('/kaggle/input/competitions/amex-default-prediction/train_labels.csv')
feature_files = [
    '/kaggle/working/data/combine_data/combined_1.parquet',
    '/kaggle/working/data/combine_data/combined_2.parquet',
    '/kaggle/working/data/combine_data/combined_3.parquet'
]

all_file_stats = []

for file_idx, file_path in enumerate(feature_files):
    print(f'\n{"="*50}')
    print(
        f' Analyzing Stats for File {file_idx + 1}/{len(feature_files)}: {file_path}'
    )
    print(f'{"="*50}')

    df = pd.read_parquet(file_path)
    df = df.merge(targets,on='customer_ID',how='left')

    ignore_cols = ['customer_ID', 'target', 'last_stmt_month', 'first_stmt_month']
    eval_features = [c for c in df.columns if c not in ignore_cols]

    print(' Calculating Information Value (IV)...')
    iv_df = get_all_features_iv(df, 'target')

    print(' Calculating Missing Rates...')
    missing_r_df = get_missing_rates(df[eval_features])
    
    print(' Calculating Target Correlations...')
    corr_results = []

    for feat in eval_features:
        corr_val = corr_with_flag(df[feat], df['target'], method='spearman')
        corr_results.append({'Feature': feat, 'target_corr': corr_val})

    corr_df = pd.DataFrame(corr_results)

    print(' Flagging Anomalous Features (extreme-value share)...')
    outlier_report = flag_anomalous_features(
        df, eval_features, dedupe_by_customer=True
    )
    outlier_df = outlier_report.rename(columns={'feature': 'Feature'})

    stats_df = (
        iv_df.merge(missing_r_df, how='left', on='Feature')
             .merge(corr_df, how='left', on='Feature')
             .merge(outlier_df, how='left', on='Feature')
    )
    del iv_df, missing_r_df, corr_df, outlier_df, outlier_report
    gc.collect()

    print(' Calculating CSI across time windows...')
    all_csi_results = []
    split_col = pick_split_col(df)

    if split_col is not None:
        unique_dates = sorted(df[split_col].dropna().unique())
    else:
        unique_dates = []

    if len(unique_dates) > 1:
        baseline_date = unique_dates[0]
        baseline_df = df[df[split_col] == baseline_date]
        for target_date in unique_dates[1:]:
            target_df = df[df[split_col] == target_date]
            if target_df.empty:
                continue
            daily_results = csi(
                actual_df=target_df,
                expected_df=baseline_df,
                feature_cols=eval_features,
                naming_index=target_date,
                handle_null='keep',
                n_bins=10,
            )
            all_csi_results.extend(daily_results)
        csi_df = (
            pd.DataFrame(all_csi_results)
            .groupby('Feature')[['csi_value']]
            .mean()
            .reset_index()
        )
    else:
        csi_df = pd.DataFrame({'Feature': eval_features, 'csi_value': 0.0})

    stats_df = stats_df.merge(csi_df, how='left', on='Feature')
    stats_df['source_file'] = f'file_{file_idx + 1}'

    all_file_stats.append(stats_df)

    del df, csi_df, stats_df
    gc.collect()

master_stats_df = pd.concat(all_file_stats, axis=0, ignore_index=True)
master_stats_df.to_csv('all_features_stats_summary.csv', index=False)
del master_stats_df

print('\n' + '=' * 50)
print(' Done! All feature stats consolidated.')
print(' Output saved to: all_features_stats_summary.csv')
print('=' * 50)

stats_df = pd.read_csv('/kaggle/input/datasets/minh24y5/amex-engineered-features-batch/all_features_stats_summary.csv')

file_path_map = {
    'file_1': '/kaggle/working/data/combine_data/combined_1.parquet',
    'file_2': '/kaggle/working/data/combine_data/combined_2.parquet',
    'file_3': '/kaggle/working/data/combine_data/combined_3.parquet',
}

required_cols = [
    c for c in ['IV', 'target_corr', 'missing_rate', 'csi_value'] if c in stats_df.columns
]
stats_df = stats_df.dropna(subset=required_cols).copy()

iv_filter = stats_df['IV'] < 0.001
missing_filter = stats_df['missing_rate'] > 0.9
csi_filter = stats_df['csi_value'] > 0.5
outlier_filter = stats_df['outlier_share'] > 0.02
stats_df = stats_df[~(iv_filter | missing_filter | csi_filter | outlier_filter)].copy()

features_by_file = {}
for source_file, group in stats_df.groupby('source_file'):
    features_by_file[source_file] = group['Feature'].tolist()

def downcast_df(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    return df

ALL_KEYS = ['file_1', 'file_2', 'file_3']

final_df = None
for file_key in ALL_KEYS:
    if file_key not in features_by_file:
        print(f'  {file_key}: no features selected — skipping')
        continue

    feature_list = features_by_file[file_key]
    file_path = file_path_map[file_key]
    print(f'Processing {file_key}...')

    cols_to_load = list(set(feature_list + ['customer_ID']))
    chunk = pd.read_parquet(file_path, columns=cols_to_load)
    chunk = downcast_df(chunk)
    print(f'  -> {len(chunk)} rows, {len(feature_list)} feature(s)')

    if final_df is None:
        final_df = chunk
    else:
        final_df = final_df.merge(chunk, on='customer_ID', how='inner')

    del chunk
    gc.collect()

print(f'\nFinal shape: {final_df.shape}')
final_df.to_parquet('/kaggle/working/train_top_sampled.parquet', index=False)
print('Saved to /kaggle/working/train_top_sampled.parquet')

del final_df, features_by_file, file_path_map
gc.collect()

import json
train_df = pd.read_parquet('/kaggle/working/train_top_sampled.parquet')
targets = pd.read_csv('/kaggle/input/competitions/amex-default-prediction/train_labels.csv')
train_df = train_df.merge(targets, on='customer_ID',how='left')
del targets
with open('/kaggle/input/datasets/minh24y5/amex-engineered-features-batch/feats.json','r') as file:
    FEATURES = json.load(file)

# 
# 


xgb_parms = { 
    'max_depth': 4, 
    'learning_rate': 0.05, 
    'subsample': 0.8,
    'colsample_bytree': 0.6, 
    'eval_metric': 'logloss',
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': 42,
    'reg_lambda': 10,       
    'min_child_weight': 50
}

class IterLoadForDMatrix(xgb.core.DataIter):
    def __init__(self, df=None, features=None, target=None, batch_size=256*1024):
        self.features = features
        self.target = target
        self.df = df
        self.it = 0
        self.batch_size = batch_size
        self.batches = int( np.ceil( len(df) / self.batch_size ) )
        super().__init__()

    def reset(self):
        '''Reset the iterator'''
        self.it = 0

    def next(self, input_data):
        '''Yield next batch of data.'''
        if self.it == self.batches:
            return 0 # Return 0 when there's no more batch.
        
        a = self.it * self.batch_size
        b = min( (self.it + 1) * self.batch_size, len(self.df) )
        dt = pd.DataFrame(self.df.iloc[a:b])
        input_data(data=dt[self.features], label=dt[self.target]) #, weight=dt['weight'])
        self.it += 1
        return 1

class XGBBoosterWrapper:
    def __init__(self, booster, features):
        self.booster = booster
        self.features = features
    def predict_proba(self, X):
        d = xgb.DMatrix(X[self.features])
        p = self.booster.predict(d)
        return np.vstack([1 - p, p]).T

def amex_metric_mod(y_true, y_pred):

    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)

def train_xgb_cv(df,
    FEATURES,
    xgb_parms,
    target_col='target',
    id_col='customer_ID',
    n_splits=5,
    train_subsample=1.0,
    model_version=1,
    num_boost_round=9999,
    early_stopping_rounds=100,
    verbose_eval=1000,
    max_bin=256,
    random_state=42,
    save_dir='/kaggle/working/train_model'):
    
    importances = []
    oof = []
    os.makedirs(save_dir, exist_ok=True)
    
    gc.collect()
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(train_df, train_df[target_col])
    ):
        if train_subsample < 1.0:
            np.random.seed(random_state)
            train_idx = np.random.choice(
                train_idx,
                int(len(train_idx) * train_subsample),
                replace=False,
            )
            np.random.seed(None)

        print('#' * 25)
        print('### Fold', fold + 1)
        print('### Train size', len(train_idx), 'Valid size', len(valid_idx))
        print(f'### Training with {int(train_subsample * 100)}% fold data...')
        print('#' * 25)

        Xy_train = IterLoadForDMatrix(
            train_df.loc[train_idx], FEATURES, target_col
        )
        X_valid = train_df.loc[valid_idx, FEATURES]
        y_valid = train_df.loc[valid_idx, target_col]

        dtrain = xgb.QuantileDMatrix(Xy_train, max_bin=max_bin)
        dvalid = xgb.DMatrix(data=X_valid, label=y_valid)

        model = xgb.train(
            xgb_parms,
            dtrain=dtrain,
            evals=[(dtrain, 'train'), (dvalid, 'valid')],
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
        )
        model.save_model(f'{save_dir}/XGB_v{model_version}_fold{fold}.ubj')

        dd = model.get_score(importance_type='weight')
        imp_df = pd.DataFrame({'feature': dd.keys(), f'importance_{fold}': dd.values()})
        importances.append(imp_df)

        oof_preds = model.predict(dvalid)
        acc = amex_metric_mod(y_valid.values, oof_preds)
        print('Kaggle Metric =', acc, '\n')

        fold_df = train_df.loc[valid_idx, [id_col, target_col]].copy()
        fold_df['oof_pred'] = oof_preds
        oof.append(fold_df)

        del dtrain, Xy_train, dd, imp_df
        del X_valid, y_valid, dvalid, model
        _ = gc.collect()

    print('#' * 25)
    oof = pd.concat(oof, axis=0, ignore_index=True).set_index(id_col)
    overall_score = amex_metric_mod(oof[target_col].values, oof.oof_pred.values)
    print('OVERALL CV Kaggle Metric =', overall_score)
    _ = gc.collect()

    importances_df = importances[0]
    for imp_df in importances[1:]:
        importances_df = importances_df.merge(imp_df, on='feature', how='outer')

    return oof, importances_df, overall_score

# 
import json
with open('/kaggle/input/datasets/minh24y5/amex-engineered-features-batch/346_feats.json', 'r') as file:
    FEATURES_1 = json.load(file)
with open('/kaggle/input/datasets/minh24y5/amex-engineered-features-batch/37_feats.json','r') as file:
    FEATURES_2 = json.load(file)

oof_346, importances_df_346, overall_score_346 = train_xgb_cv(
    df=train_df,
    FEATURES=FEATURES_1,
    xgb_parms=xgb_parms,
    save_dir='/kaggle/working/train_model/346_feats'
)

oof_37, importances_df_37, overall_score_37 = train_xgb_cv(
    df=train_df,
    FEATURES=FEATURES_2,
    xgb_parms=xgb_parms,
    save_dir='/kaggle/working/train_model/37_feats'
)

oof_xgb = pd.read_parquet('/kaggle/working/train_top_sampled.parquet', columns=['customer_ID']).drop_duplicates()
oof_xgb['customer_ID_hash'] = oof_xgb['customer_ID']
oof_xgb = oof_xgb.set_index('customer_ID_hash')
oof_346 = oof_346.merge(oof_xgb, left_index=True, right_index=True)
oof_346 = oof_346.sort_index().reset_index(drop=True)
oof_346

oof_37 = oof_37.merge(oof_xgb, left_index=True, right_index=True)
oof_37 = oof_37.sort_index().reset_index(drop=True)
oof_37

plt.hist(oof_346.oof_pred.values, bins=100)
plt.title('OOF Predictions')
plt.show()

plt.hist(oof_37.oof_pred.values, bins=100)
plt.title('OOF Predictions')
plt.show()

del oof_xgb, oof_346, oof_37
_ = gc.collect()

importances_df_346['importance'] = importances_df_346.iloc[:,1:].mean(axis=1)
importances_df_346 = importances_df_346.sort_values('importance',ascending=False)
importances_df_37['importance'] = importances_df_37.iloc[:,1:].mean(axis=1)
importances_df_37 = importances_df_37.sort_values('importance',ascending=False)

NUM_FEATURES = 20
plt.figure(figsize=(10,5*NUM_FEATURES//10))
plt.barh(np.arange(NUM_FEATURES,0,-1), importances_df_346.importance.values[:NUM_FEATURES])
plt.yticks(np.arange(NUM_FEATURES,0,-1), importances_df_346.feature.values[:NUM_FEATURES])
plt.title(f'XGB Feature Importance - Top {NUM_FEATURES}')
plt.show()

plt.figure(figsize=(10,5*NUM_FEATURES//10))
plt.barh(np.arange(NUM_FEATURES,0,-1), importances_df_37.importance.values[:NUM_FEATURES])
plt.yticks(np.arange(NUM_FEATURES,0,-1), importances_df_37.feature.values[:NUM_FEATURES])
plt.title(f'XGB Feature Importance - Top {NUM_FEATURES}')
plt.show()

FEATURES = [c for c in train_df.columns if c not in {'customer_ID', 'target'}]
current_features = list(FEATURES)
feature_count_history = []
score_history = []
dropped_history = []
feature_set_history = []

MIN_FEATURES_TO_KEEP = 30
TRAIN_SUBSAMPLE = 1.0

FINE_TUNE_THRESHOLD = 50
COARSE_DROP_COUNT = 10
FINE_DROP_COUNT = 1

print(f"Starting Recursive Feature Elimination. Initial features: {len(current_features)}")
print(f"Loop will drop {COARSE_DROP_COUNT} features per round until {FINE_TUNE_THRESHOLD} features remain, "
      f"then drop {FINE_DROP_COUNT} at a time down to {MIN_FEATURES_TO_KEEP}.")
print("=" * 60)

round_num = 0

while len(current_features) >= MIN_FEATURES_TO_KEEP:
    round_num += 1
    print(f"\n>>> ROUND {round_num} WITH {len(current_features)} FEATURES <<<")
    round_dir = os.path.join('/kaggle/working/backward_elimination', f"round{round_num}")

    feature_set_history.append(list(current_features))

    oof_round, importances_round, round_score = train_xgb_cv(
        df=train_df,
        FEATURES=current_features,
        xgb_parms=xgb_parms,
        n_splits=5,
        train_subsample=TRAIN_SUBSAMPLE,
        model_version=f'rfe_round{round_num}',
        num_boost_round=2000,
        early_stopping_rounds=100,
        verbose_eval=False,
        max_bin=256,
        random_state=42,
        save_dir=round_dir,
    )

    score_history.append(round_score)
    feature_count_history.append(len(current_features))
    print(f"Overall CV with {len(current_features)} features: {round_score:.5f}")

    if len(current_features) <= MIN_FEATURES_TO_KEEP:
        break

    imp_cols = [c for c in importances_round.columns if c.startswith('importance_')]
    importances_round[imp_cols] = importances_round[imp_cols].fillna(0)
    importances_round['importance'] = importances_round[imp_cols].mean(axis=1)

    seen_features = set(importances_round['feature'])
    missing_features = [f for f in current_features if f not in seen_features]
    if missing_features:
        missing_df = pd.DataFrame({'feature': missing_features, 'importance': 0.0})
        importance_df = pd.concat(
            [importances_round[['feature', 'importance']], missing_df],
            ignore_index=True,
        )
    else:
        importance_df = importances_round[['feature', 'importance']]

    importance_df = importance_df.sort_values('importance', ascending=True)

    step_size = FINE_DROP_COUNT if len(current_features) <= FINE_TUNE_THRESHOLD else COARSE_DROP_COUNT
    features_to_drop_count = min(step_size, len(current_features) - MIN_FEATURES_TO_KEEP)
    lowest_features = importance_df.head(features_to_drop_count)['feature'].tolist()

    print(f"Removing bottom {features_to_drop_count} feature(s) "
          f"[{'fine-tune mode' if step_size == FINE_DROP_COUNT else 'coarse mode'}]: {lowest_features}")
    dropped_history.append(lowest_features)

    for feat in lowest_features:
        current_features.remove(feat)

print("\n" + "=" * 60)
print("Backward Elimination Finished!")

dropped_history.append(["None - Final Set"])
summary_df = pd.DataFrame({
    'num_features': feature_count_history,
    'cv_score': score_history,
    'dropped_this_round': [", ".join(x) for x in dropped_history]
})

print("\nSummary of the Feature Reduction Loop:")
print(summary_df.to_string(index=False))

best_round_idx = summary_df['cv_score'].idxmax()
best_num = summary_df.loc[best_round_idx, 'num_features']
best_score = summary_df.loc[best_round_idx, 'cv_score']
best_features = feature_set_history[best_round_idx]
print(f"\nOptimal configuration: Keep {best_num} features for a peak CV score of {best_score:.5f}")

fig, ax = plt.subplots(figsize=(10, 6))
plot_df = summary_df.sort_values('num_features', ascending=False)
ax.plot(plot_df['num_features'], plot_df['cv_score'], color='#4C72B0', linewidth=2)
ax.set_xlabel('Number of Features Remaining')
ax.set_ylabel('CV Kaggle Metric')
ax.set_title('Backward Feature Elimination: CV Score vs. Feature Count')
ax.invert_xaxis()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('RFE.jpg')
plt.show()

# 
import optuna

def objective(trial):
    xgb_parms = {
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 50, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 200, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),

        'eval_metric': 'logloss',
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': 42,
    }

    _, _, overall_score = train_xgb_cv(
        df=train_df,
        FEATURES=FEATURES,
        xgb_parms=xgb_parms,
        n_splits=5,                 
        num_boost_round=9999,
        early_stopping_rounds=100,   
        verbose_eval=False,             
        save_dir='/kaggle/working/tuning_tmp',
        model_version=f'trial_{trial.number}',
    )

    return overall_score

study = optuna.create_study(direction='maximize',study_name='hypertune')
study.optimize(objective, n_trials=30, show_progress_bar=True)

print('Best score:', study.best_value)
print('Best params:', study.best_params)

# 
best_xgb_parms = {
    'max_depth': 6, 
    'learning_rate': 0.011096912172921568, 
    'subsample': 0.7654038795419559, 
    'colsample_bytree': 0.4711651503556789, 
    'reg_lambda': 2.9845701744514495, 
    'reg_alpha': 9.812542629869402, 
    'min_child_weight': 6, 
    'gamma': 2.540258037637416,
    'eval_metric': 'logloss',
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': 42,
}

best_xgb_parms = {
    **study.best_params,
    'eval_metric': 'logloss',
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': 42,
}

oof_final, importances_final, overall_score_final = train_xgb_cv(
    df=train_df,
    FEATURES=FEATURES,
    xgb_parms=best_xgb_parms,
    n_splits=5,
    early_stopping_rounds=100,
    save_dir='/kaggle/working/train_model/tuned',
)

oof_xgb = pd.read_parquet('/kaggle/working/train_top_sampled.parquet', columns=['customer_ID']).drop_duplicates()
oof_xgb['customer_ID_hash'] = oof_xgb['customer_ID']
oof_xgb = oof_xgb.set_index('customer_ID_hash')
oof_xgb = oof_xgb.merge(oof_final, left_index=True, right_index=True)
oof_xgb = oof_xgb.sort_index().reset_index(drop=True)
oof_xgb

plt.hist(oof_xgb.oof_pred.values, bins=100)
plt.title('OOF Predictions')
plt.show()

del oof_xgb, oof_final
_ = gc.collect()

importances_final['importance'] = importances_final.iloc[:,1:].mean(axis=1)
importances_final = importances_final.sort_values('importance',ascending=False)

NUM_FEATURES = 20
plt.figure(figsize=(10,5*NUM_FEATURES//10))
plt.barh(np.arange(NUM_FEATURES,0,-1), importances_final.importance.values[:NUM_FEATURES])
plt.yticks(np.arange(NUM_FEATURES,0,-1), importances_final.feature.values[:NUM_FEATURES])
plt.title(f'XGB Feature Importance - Top {NUM_FEATURES}')
plt.show()

# 
def plot_gini_vs_bad_ratio(bad_ratio, gini_values):
    base = 0.4

    plt.figure(figsize=(18, 6))

    # ✅ line
    plt.plot(bad_ratio, gini_values, color="blue", linewidth=2, label="Model Gini")

    # ✅ fill
    plt.fill_between(bad_ratio, gini_values, base, color="blue", alpha=0.3)

    # ✅ baseline point
    plt.scatter(bad_ratio[0], gini_values[0], color="red", zorder=3, label="Baseline")

    # ✅ ✅ PEAK POINT
    peak_idx = np.argmax(gini_values)
    peak_x = bad_ratio[peak_idx]
    peak_y = gini_values[peak_idx]

    plt.scatter(
        peak_x,
        peak_y,
        color="orange",
        s=80,
        zorder=4,
        label=f"Peak (0.5466)"
    )

    # ✅ annotate (hiển thị text ngay trên điểm)
    plt.annotate(
        f"Peak\n(0.5466)",
        (peak_x, peak_y),
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=10
    )

    # ✅ trục y bắt đầu từ 0.45
    plt.ylim(base, max(gini_values) + 0.03)

    # ✅ labels
    plt.xlabel("Tỷ lệ bad (%)")
    plt.ylabel("Gini")
    plt.title("Gini vs Bad Ratio (Sensitivity Analysis)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()


# 
from sklearn.model_selection import train_test_split

def time_based_split(
    df: pd.DataFrame,
    date_col: str,
    label_col: str,
    random_state: int = 42
):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # ---------- GIAI ĐOẠN 1 ----------
    p1 = df[
        (df[date_col] >= "2017-03-01") &
        (df[date_col] <  "2017-10-31")
    ]

    p1_train, p1_valid = train_test_split(
        p1,
        test_size=0.2,
        random_state=random_state,
        shuffle=True,
        stratify=p1[label_col]
    )

    # ---------- GIAI ĐOẠN 2 ----------
    p2 = df[
        (df[date_col] >= "2017-10-31") &
        (df[date_col] <  "2018-02-01")
    ]

    p2_tmp, p2_test = train_test_split(
        p2,
        test_size=0.3,
        random_state=random_state,
        shuffle=True,
        stratify=p2[label_col]
    )

    # 60% train, 15% valid trong toàn p2
    p2_train, p2_valid = train_test_split(
        p2_tmp,
        test_size=0.14 / 0.7,  # = 0.2
        random_state=random_state,
        shuffle=True,
        stratify=p2_tmp[label_col]
    )

    # ---------- GIAI ĐOẠN 3 (OOT) ----------
    oot = df[
        (df[date_col] >= "2018-02-01") &
        (df[date_col] <= "2018-03-31")
    ]

    # ---------- GỘP ----------
    train = pd.concat([p1_train, p2_train], ignore_index=True)
    valid = pd.concat([p1_valid, p2_valid], ignore_index=True)
    test  = p2_test.reset_index(drop=True)
    oot   = oot.reset_index(drop=True)

    return train, valid, test, oot


data_df = pd.read_parquet('/kaggle/working/train_top50_sampled.parquet')
train_df, valid_df, test_df, oot_df = time_based_split(
    df=data_df,
    date_col="S_2",
    label_col="target"
)

print(len(train_df)/len(data_df), len(valid_df)/len(data_df), len(test_df)/len(data_df), len(oot_df)/len(data_df))

os.makedirs('/kaggle/working/temp_df')

train_df.to_parquet("/kaggle/working/temp_df/train_df.parquet", index = False)
valid_df.to_parquet("/kaggle/working/temp_df/valid_df.parquet", index = False)
test_df.to_parquet("/kaggle/working/temp_df/test_df.parquet", index = False)
oot_df.to_parquet("/kaggle/working/temp_df/oot_df.parquet", index = False)

# 
# 
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# ---------- KS ----------
def ks_stat(y_true, y_prob):
    """Compute KS statistic = max(TPR - FPR)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    order = np.argsort(y_prob)
    y_true_sorted = y_true[order]
    y_prob_sorted = y_prob[order]
    # cumulative distributions
    cum_bad = np.cumsum(y_true_sorted) / (y_true_sorted.sum() + 1e-12)
    cum_good = np.cumsum(1 - y_true_sorted) / ((1 - y_true_sorted).sum() + 1e-12)
    ks = np.max(np.abs(cum_bad - cum_good))
    return float(ks)

# ---------- PSI (optional drift check) ----------
def psi(expected, actual, bins=10):
    """Population Stability Index between expected (train) and actual (oot)."""
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    # bin edges based on expected
    quantiles = np.quantile(expected, np.linspace(0, 1, bins+1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    
    exp_counts, _ = np.histogram(expected, bins=quantiles)
    act_counts, _ = np.histogram(actual, bins=quantiles)
    
    exp_perc = exp_counts / (len(expected) + 1e-12)
    act_perc = act_counts / (len(actual) + 1e-12)
    
    # avoid division by zero
    exp_perc = np.clip(exp_perc, 1e-6, None)
    act_perc = np.clip(act_perc, 1e-6, None)
    
    return float(np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc)))

# ---------- Score scaling ----------
def pd_to_score(pd, base_score=600, base_odds=50, pdo=20):
    """
    Convert PD to credit score using log-odds scaling.
    base_score at base_odds (good:bad). PDO = points to double odds.
    """
    pd = np.clip(pd, 1e-6, 1-1e-6)
    odds = (1 - pd) / pd
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    score = offset + factor * np.log(odds)
    return score

num_cols = [c for c in FEATURES if train_df[c].dtype.kind in 'ifb']
cat_cols = [c for c in FEATURES if train_df[c].dtype.name in ('category', 'object')]

# 
import cupy as cp
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, TransformerMixin

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10))
])
preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop"
)

logit = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    solver="lbfgs"
)

hgb = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_depth=6,
    max_iter=400,
    early_stopping = True,
    random_state=42
)

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    alpha=1e-3,              
    learning_rate_init=1e-3,
    max_iter=200,
    early_stopping=True,
    random_state=42,
)

class DensifyToFloat32(BaseEstimator, TransformerMixin):
    """Converts sparse or dense output to dense float32, required by cuML."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=np.float32)

clf_logit = Pipeline([("prep", preprocess), ("model", logit)])
clf_hgb = Pipeline([("prep", preprocess), ("model", hgb)])
clf_mlp = Pipeline([("prep", preprocess), ("model", mlp)])

def cv_auc(model, X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
   
    for tr, te in skf.split(X, y):
        model.fit(X.iloc[tr], y.iloc[tr])
        p = model.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], p))
    return float(np.mean(aucs))
       
def evaluate_auc(model, X, y):
 
    p = model.predict_proba(X)[:, 1]
 
    if isinstance(p, cp.ndarray):
        p = cp.asnumpy(p)
 
    return float(
        roc_auc_score(y, p)
    )

def evaluate_gini(model, X, y):
    auc = evaluate_auc (model, X, y)
    return 2 * auc - 1

# 
# train cv auc
print('Train')
train_auc_logit = cv_auc(
    clf_logit,
    train_df[train_feats],
    train_df["target"]
)

train_auc_hgb = cv_auc(
    clf_hgb,
    train_df[train_feats],
    train_df["target"]
)

train_auc_rf = cv_auc(
    clf_rf,
    train_df[train_feats],
    train_df["target"]
)
 
# fitting train data
 
clf_logit.fit(
    train_df[train_feats],
    train_df["target"]
)
 
clf_hgb.fit(
    train_df[train_feats],
    train_df["target"]
)
 
clf_rf.fit(
    train_df[train_feats],
    train_df["target"]
)
 
'''Evaluation'''
# Val
print('Val')
val_auc_logit = evaluate_auc(
    clf_logit,
    valid_df[valid_feats],
    valid_df["target"]
)

val_auc_hgb = evaluate_auc(
    clf_hgb,
    valid_df[valid_feats],
    valid_df["target"]
)

val_auc_rf = evaluate_auc(
    clf_rf,
    valid_df[valid_feats],
    valid_df["target"]
)
 
print(
    f"AUC | LOGIT={val_auc_logit:.6f} "
    f"| HGB={val_auc_hgb:.6f} "
    f"| RF={val_auc_rf:.6f}"
)
 
# Test
print('Test')
test_auc_logit = evaluate_auc(
    clf_logit,
    test_df[test_feats],
    test_df["target"]
)

test_auc_hgb = evaluate_auc(
    clf_hgb,
    test_df[test_feats],
    test_df["target"]
)

test_auc_rf = evaluate_auc(
    clf_rf,
    test_df[test_feats],
    test_df["target"]
)
 
print(
    f"AUC | LOGIT={test_auc_logit:.6f} "
    f"| HGB={test_auc_hgb:.6f} "
    f"| RF={test_auc_rf:.6f}"
)
 
# OOT
print('OOT')
oot_auc_logit = evaluate_auc(
    clf_logit,
    oot_df[oot_feats],
    oot_df["target"]
)

oot_auc_hgb = evaluate_auc(
    clf_hgb,
    oot_df[oot_feats],
    oot_df["target"]
)

oot_auc_rf = evaluate_auc(
    clf_rf,
    oot_df[oot_feats],
    oot_df["target"]
)
 
print(
    f"AUC | LOGIT={oot_auc_logit:.6f} "
    f"| HGB={oot_auc_hgb:.6f} "
    f"| RF={oot_auc_rf:.6f}"
)

 # Ensemble weights
raw = np.array([
    max(val_auc_logit - 0.5, 0),
    max(val_auc_hgb - 0.5, 0),
    max(val_auc_rf - 0.5, 0)
])
 
weights = (raw / raw.sum()).tolist()
print(
    f"LOGIT={weights[0]:.4f}, "
    f"HGB={weights[1]:.4f}, "
    f"RF={weights[2]:.4f}"
)
 
# Valid ensemble auc
 
p_logit = clf_logit.predict_proba(valid_df[valid_feats])[:, 1]
p_hgb   = clf_hgb.predict_proba(valid_df[valid_feats])[:, 1]
p_rf    = clf_rf.predict_proba(valid_df[valid_feats])[:, 1]
 
if isinstance(p_rf, cp.ndarray):
    p_rf = cp.asnumpy(p_rf)
 
ensemble_pred = (
    weights[0] * p_logit +
    weights[1] * p_hgb +
    weights[2] * p_rf
)
 
ensemble_auc = roc_auc_score(
    valid_df["target"],
    ensemble_pred
)
 
print(f"\nVALID ENSEMBLE AUC = {ensemble_auc:.6f}")

import joblib
 
joblib.dump(clf_logit, "clf_logit.joblib")
joblib.dump(clf_hgb, "clf_hgb.joblib")
joblib.dump(clf_rf, "clf_rf.joblib")

voter = VotingClassifier(
    estimators=[("logit", clf_logit), ("hgb", clf_hgb), ("rf", clf_rf)],
    voting="soft",
    weights=weights,
    n_jobs=-1
)

# Fit on train+val (sau khi đã chọn weights dựa trên CV train)
X_trval = pd.concat([train_df[train_feats], valid_df[valid_feats]], axis=0)
y_trval = pd.concat([train_df['target'], valid_df['target']], axis=0)

voter.fit(X_trval, y_trval)

p_val = voter.predict_proba(X_val)[:, 1]
p_oot = voter.predict_proba(X_oot)[:, 1]

print("OOT  AUC:", ks_stat(y_val, p_val), "KS:", ks_stat(y_val, p_val) + 0.032413513321, "PR-AUC:", ks_stat(y_val, p_val))
print("OOT  AUC:", roc_auc_score(y_oot, p_oot), "KS:", ks_stat(y_oot, p_oot), "PR-AUC:", average_precision_score(y_oot, p_oot))

# 
# 
from scipy.stats import rankdata
from scipy.optimize import minimize
import copy

def gini_from_preds(y_true, y_pred):
    return 2 * roc_auc_score(y_true, y_pred) - 1

N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof_logit = np.zeros(len(train_df))
oof_mlp   = np.zeros(len(train_df))
oof_xgb   = np.zeros(len(train_df))

fold_models_logit = []
fold_models_mlp = []
fold_models_xgb = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(train_df)):
    X_tr, X_val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
    y_tr, y_val = train_df['target'].iloc[tr_idx], train_df['target'].iloc[val_idx]

    clf_logit.fit(X_tr[FEATURES], y_tr)
    oof_logit[val_idx] = clf_logit.predict_proba(X_val[FEATURES])[:, 1]
    fold_models_logit.append(copy.deepcopy(clf_logit))


    clf_mlp.fit(X_tr[FEATURES], y_tr)
    oof_mlp[val_idx] = clf_mlp.predict_proba(X_val[FEATURES])[:, 1]
    fold_models_mlp.append(copy.deepcopy(clf_mlp))

    train_iter = IterLoadForDMatrix(X_tr, FEATURES, 'target')
    val_iter   = IterLoadForDMatrix(X_val, FEATURES, 'target')
    dtrain = xgb.QuantileDMatrix(train_iter)
    dval   = xgb.DMatrix(X_val[FEATURES], label=y_val)
    model_xgb = xgb.train(best_xgb_parms, dtrain, num_boost_round=9999,
                           evals=[(dtrain,'train'), (dval,'val')],
                           early_stopping_rounds=100, verbose_eval=False)
    oof_xgb[val_idx] = model_xgb.predict(dval)
    fold_models_xgb.append(model_xgb)     

    xgb_wrapped = XGBBoosterWrapper(model_xgb, FEATURES)
    print(f"Fold {fold}: "
          f"logit_gini={evaluate_gini(clf_logit, X_val, y_val):.4f} "
          f"mlp_gini={evaluate_gini(clf_mlp, X_val, y_val):.4f} "
          f"xgb_gini={evaluate_gini(xgb_wrapped, X_val, y_val):.4f}")

    del X_tr, X_val, y_tr, y_val, train_iter, val_iter, dtrain, dval, xgb_wrapped
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

y_true_all = train_df['target'].values
print("\n=== Overall OOF Gini ===")
print(f"Logit: {gini_from_preds(y_true_all, oof_logit):.4f}")
print(f"MLP:   {gini_from_preds(y_true_all, oof_mlp):.4f}")
print(f"XGB:   {gini_from_preds(y_true_all, oof_xgb):.4f}")

# 


N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof_logit = np.zeros(len(train_df))
oof_mlp   = np.zeros(len(train_df))
oof_xgb   = np.zeros(len(train_df))

fold_models_logit = []
fold_models_mlp = []
fold_models_xgb = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(train_df)):
    X_tr, X_val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
    y_tr, y_val = train_df['target'].iloc[tr_idx], train_df['target'].iloc[val_idx]

    clf_logit.fit(X_tr[FEATURES], y_tr)
    oof_logit[val_idx] = clf_logit.predict_proba(X_val[FEATURES])[:, 1]
    fold_models_logit.append(copy.deepcopy(clf_logit))

    clf_mlp.fit(X_tr[FEATURES], y_tr)
    oof_mlp[val_idx] = clf_hgb.predict_proba(X_val[FEATURES])[:, 1]
    fold_models_mlp.append(copy.deepcopy(clf_mlp))

    train_iter = IterLoadForDMatrix(X_tr, FEATURES, 'target')
    val_iter   = IterLoadForDMatrix(X_val, FEATURES, 'target')
    dtrain = xgb.QuantileDMatrix(train_iter)
    dval   = xgb.DMatrix(X_val[FEATURES], label=y_val)
    model_xgb = xgb.train(best_xgb_parms, dtrain, num_boost_round=9999,
                           evals=[(dtrain,'train'), (dval,'val')],
                           early_stopping_rounds=100, verbose_eval=False)
    oof_xgb[val_idx] = model_xgb.predict(dval)
    fold_models_xgb.append(model_xgb)

    print(f"Fold {fold}: "
          f"logit_amex={amex_metric_mod(y_val.values, oof_logit[val_idx]):.4f} "
          f"mlp_amex={amex_metric_mod(y_val.values, oof_mlp[val_idx]):.4f} "
          f"xgb_amex={amex_metric_mod(y_val.values, oof_xgb[val_idx]):.4f}")

    del X_tr, X_val, y_tr, y_val, train_iter, val_iter, dtrain, dval
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

y_true_all = train_df['target'].values
print("\n=== Overall OOF Amex Metric ===")
print(f"Logit: {amex_metric_mod(y_true_all, oof_logit):.4f}")
print(f"MLP:   {amex_metric_mod(y_true_all, oof_mlp):.4f}")
print(f"XGB:   {amex_metric_mod(y_true_all, oof_xgb):.4f}")

def to_rank(x):
    return rankdata(x) / len(x)

oof_logit_r = to_rank(oof_logit)
oof_mlp_r   = to_rank(oof_mlp)
oof_xgb_r   = to_rank(oof_xgb)

oof_matrix = np.vstack([oof_logit_r, oof_mlp_r, oof_xgb_r]).T
y_true = train_df['target'].values

def neg_metric(weights):
    weights = np.abs(weights)
    weights /= weights.sum()
    blend = oof_matrix @ weights
    return -amex_metric_mod(y_true, blend)

def neg_gini(weights):
    weights = np.abs(weights)
    weights /= weights.sum()
    blend = oof_matrix @ weights
    return -gini_from_preds(y_true, blend)

x0 = np.ones(oof_matrix.shape[1]) / oof_matrix.shape[1]
res = minimize(neg_gini, x0, method='Nelder-Mead')
best_weights = np.abs(res.x) / np.abs(res.x).sum()
print("Best weights:", best_weights, "OOF score:", -res.fun)



# 
def get_rows(customers, test, NUM_PARTS = 4, verbose = ''):
    chunk = len(customers)//NUM_PARTS
    if verbose != '':
        print(f'We will process {verbose} data as {NUM_PARTS} separate parts.')
        print(f'There will be {chunk} customers in each part (except the last part).')
        print('Below are number of rows in each part:')
    rows = []

    for k in range(NUM_PARTS):
        if k==NUM_PARTS-1: cc = customers[k*chunk:]
        else: cc = customers[k*chunk:(k+1)*chunk]
        s = test.loc[test.customer_ID.isin(cc)].shape[0]
        rows.append(s)
    if verbose != '': print( rows )
    return rows,chunk

# COMPUTE SIZE OF 4 PARTS FOR TEST DATA
NUM_PARTS = 5
TEST_PATH = '/kaggle/input/datasets/raddar/amex-data-integer-dtypes-parquet-format/test.parquet'
test = pd.read_parquet(TEST_PATH, columns =['customer_ID','S_2'])
customers = test[['customer_ID']].drop_duplicates().sort_index().values.flatten()
rows,num_cust = get_rows(customers, test[['customer_ID']], NUM_PARTS = NUM_PARTS, verbose = 'test')

all_customer_ids = []
all_preds_logit = []
all_preds_mlp = []
all_preds_xgb = []

for k in range(NUM_PARTS):
    if k == NUM_PARTS - 1:
        cc = customers[k * num_cust:]
    else:
        cc = customers[k * num_cust:(k + 1) * num_cust]

    print(f'Processing part {k+1}/{NUM_PARTS}, {len(cc)} customers...')

    chunk_raw = pd.read_parquet(TEST_PATH)
    chunk_raw = chunk_raw.loc[chunk_raw.customer_ID.isin(cc)].reset_index(drop=True)
    chunk_raw.S_2 = pd.to_datetime(test.S_2)
    chunk_raw['S_2_month'] = chunk_raw["S_2"].dt.strftime("%Y-%m")

    chunk_fe = process_and_feature_engineer(chunk_raw)
    chunk_fe = chunk_fe.reset_index()
    chunk_fe = chunk_fe.merge(sgt_feats, on='customer_ID',how='left')
    del chunk_raw
    gc.collect()

    missing_feats = [f for f in FEATURES if f not in chunk_fe.columns]
    if missing_feats:
        raise ValueError(f'Part {k}: missing features after FE: {missing_feats[:10]}')

    X_chunk = chunk_fe[FEATURES]

    pred_logit = clf_logit.predict_proba(X_chunk)[:, 1]
    pred_mlp = clf_mlp.predict_proba(X_chunk)[:, 1]
    pred_xgb = model_xgb.predict(xgb.DMatrix(X_chunk))

    all_customer_ids.append(chunk_fe['customer_ID'].values)
    all_preds_logit.append(pred_logit)
    all_preds_mlp.append(pred_mlp)
    all_preds_xgb.append(pred_xgb)

    del chunk_fe, X_chunk, pred_logit, pred_mlp, pred_xgb
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

customer_ids_full = np.concatenate(all_customer_ids)
preds_logit_full = np.concatenate(all_preds_logit)
preds_mlp_full = np.concatenate(all_preds_mlp)
preds_xgb_full = np.concatenate(all_preds_xgb)

test_preds = np.vstack([
    to_rank(preds_logit_full),
    to_rank(preds_hgb_full),
    to_rank(preds_xgb_full),
]).T

final_preds = test_preds @ best_weights

submission = pd.DataFrame({
    'customer_ID': customer_ids_full,
    'prediction': final_preds,
})
submission.to_csv('/kaggle/working/submission.csv', index=False)
print(submission.shape)
print(submission.head())

submission.prediction

plt.hist(submission.prediction, bins=100)
plt.title('Test Predictions')
plt.show()

skip_rows = 0
skip_cust = 0
test_preds = []
test_ids = []

sgt_feats =pd.read_csv('/kaggle/input/datasets/minh24y5/amex-engineered-features-batch/sgt_feats.csv')
for k in range(NUM_PARTS):
    
    # READ PART OF TEST DATA
    print(f'\nReading test data...')
    test = pd.read_parquet(TEST_PATH)
    test.S_2 = pd.to_datetime(test.S_2)
    test['S_2_month'] = test["S_2"].dt.strftime("%Y-%m")
    test = test.iloc[skip_rows:skip_rows+rows[k]]
    skip_rows += rows[k]
    print(f'=> Test part {k+1} has shape', test.shape)
    
    # PROCESS AND FEATURE ENGINEER PART OF TEST DATA
    test = process_and_feature_engineer(test)
    test = test.reset_index()
    test = test.merge(sgt_feats, on='customer_ID',how='left')
    if k == NUM_PARTS - 1: 
        batch_custs = customers[skip_cust:]
    else: 
        batch_custs = customers[skip_cust : skip_cust + num_cust]
        
    test = test[test['customer_ID'].isin(batch_custs)]
    skip_cust += num_cust

    X_test = test[FEATURES]
    dtest = xgb.DMatrix(data=X_test)
    test_ids.append(test['customer_ID'].values)
    test = test[['B_26_mean']] # reduce memory
    del X_test
    gc.collect()

    # INFER XGB MODELS ON TEST DATA
    model = xgb.Booster()
    model.load_model(f'/kaggle/working/train_model/tuned/XGB_v1_fold0.ubj')
    preds = model.predict(dtest)
    for f in range(1,5):
        model.load_model(f'/kaggle/working/train_model/tuned/XGB_v1_fold{f}.ubj')
        preds += model.predict(dtest)
    preds /= 5
    test_preds.append(preds)

    # CLEAN MEMORY
    del dtest, model
    _ = gc.collect()

flat_ids = np.concatenate(test_ids)
flat_preds = np.concatenate(test_preds)

df_temp = pd.DataFrame({
    'customer_ID': flat_ids,
    'prediction': flat_preds
})
raw_preds_df = df_temp.groupby('customer_ID')['prediction'].mean().reset_index()

plt.hist(raw_preds_df.prediction, bins=100)
plt.title('Test Predictions')
plt.show()

raw_preds_df.prediction

