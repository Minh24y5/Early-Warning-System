import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt, gc, os

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


def get_all_features_iv(df, target, id_col, max_bins=10, unique_threshold=10):
    iv_results = {}
    cols_to_skip = [target, id_col, "last_stmt_month", "first_stmt_month"]
    
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

def _create_summary(targets, feature_files, id_col, target_col):
    
    all_file_stats = []

    for file_idx, file_path in enumerate(feature_files):
        print(f'\n{"="*50}')
        print(
            f' Analyzing Stats for File {file_idx + 1}/{len(feature_files)}: {file_path}'
        )
        print(f'{"="*50}')

        df = pd.read_parquet(file_path)
        df = df.merge(targets,on=id_col,how='left')

        ignore_cols = [id_col, target_col, 'last_stmt_month', 'first_stmt_month']
        eval_features = [c for c in df.columns if c not in ignore_cols]

        print(' Calculating Information Value (IV)...')
        iv_df = get_all_features_iv(df, target_col)

        print(' Calculating Missing Rates...')
        missing_r_df = get_missing_rates(df[eval_features])
        
        print(' Calculating Target Correlations...')
        corr_results = []

        for feat in eval_features:
            corr_val = corr_with_flag(df[feat], df[target_col], method='spearman')
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

    print('\n' + '=' * 50)
    print(' Done! All feature stats consolidated.')
    print('=' * 50)

    return master_stats_df

def _merging(stats_df, file_path_map, id_col):
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

    ALL_KEYS = list(file_path_map.keys())

    final_df = None
    for file_key in ALL_KEYS:
        if file_key not in features_by_file:
            print(f'  {file_key}: no features selected — skipping')
            continue

        feature_list = features_by_file[file_key]
        file_path = file_path_map[file_key]
        print(f'Processing {file_key}...')

        cols_to_load = list(set(feature_list + [id_col]))
        chunk = pd.read_parquet(file_path, columns=cols_to_load)
        chunk = downcast_df(chunk)
        print(f'  -> {len(chunk)} rows, {len(feature_list)} feature(s)')

        if final_df is None:
            final_df = chunk
        else:
            final_df = final_df.merge(chunk, on=id_col, how='inner')

    print(f'\nFinal shape: {final_df.shape}')

    return final_df
