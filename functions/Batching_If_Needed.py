import numpy as np 
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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

def process_in_customer_chunks(df, sgt_feats=None, id_col, chunk_size=229500, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    unique_custs = df[id_col].unique()
    total_custs = len(unique_custs)
    num_chunks = int(np.ceil(total_custs / chunk_size))
    print(
        f"Total Customers: {total_custs} | Processing in {num_chunks} chunk(s) of ~{chunk_size} customers each."
    )
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_custs)
        batch_cust_ids = set(unique_custs[start_idx:end_idx])
        df_chunk = df[df[id_col].isin(batch_cust_ids)].copy()
        processed_chunk = process_and_feature_engineer(df_chunk)

        if sgt_feats:
            sgt_chunk = sgt_feats[sgt_feats[id_col].isin(batch_cust_ids)]
            processed_chunk = processed_chunk.merge(sgt_chunk, on=id_col, how='left')

        processed_chunk = optimize_dtypes(processed_chunk)
        chunk_file = os.path.join(save_dir, f"batch_{i+1}_of_{num_chunks}.parquet")
        processed_chunk.to_parquet(
            chunk_file, compression="zstd", index=False
        )
        print(f"Saved: {chunk_file}")
        del df_chunk, processed_chunk, sgt_chunk, batch_cust_ids
        gc.collect()
    print("All chunks processed successfully!")


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
    batch_files = _list_batch_files(batch _dir)
 
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

def saving_combined_batch(batch_cols, id_cols, n_files, batch_dir, output_dir):
    feature_cols_all = [c for c in batch_cols if c not in id_cols]

    print(f'Total columns: {len(batch_cols)} | Feature columns (excl. IDs): {len(feature_cols_all)}')

    col_chunks = np.array_split(feature_cols_all, n_files)

    for i, cols in enumerate(col_chunks):
        print(f'File {i+1}: {len(cols)} feature columns')
        combine_batches_select_features(
            batch_dir,
            output_dir,
            feature_cols=list(cols),
            id_cols = id_cols,
            output_filename=f'combined_{i+1}.parquet'
        )