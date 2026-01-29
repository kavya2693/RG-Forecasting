"""
Prepare data for Vertex AI training by combining shards and splitting TRAIN/VAL
"""

import pandas as pd
from google.cloud import storage
import os
import glob

def prepare_vertex_data():
    """Combine sharded exports and split into TRAIN/VAL files."""

    bucket_name = "myforecastingsales-data"
    input_prefix = "training_data/f1_full/"

    print("Downloading and combining data shards...")

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # List all shards
    blobs = list(bucket.list_blobs(prefix=input_prefix))
    csv_blobs = [b for b in blobs if b.name.endswith('.csv')]

    print(f"Found {len(csv_blobs)} CSV files")

    # Download and combine
    os.makedirs('/tmp/vertex_data', exist_ok=True)

    dfs = []
    for i, blob in enumerate(csv_blobs):
        local_path = f'/tmp/vertex_data/shard_{i:03d}.csv'
        blob.download_to_filename(local_path)
        df = pd.read_csv(local_path)
        dfs.append(df)

        if (i + 1) % 20 == 0:
            print(f"  Downloaded {i + 1}/{len(csv_blobs)} files...")

    print("Combining all shards...")
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(full_df):,}")

    # Split by split_role
    train_df = full_df[full_df['split_role'] == 'TRAIN']
    val_df = full_df[full_df['split_role'] == 'VAL']

    print(f"TRAIN rows: {len(train_df):,}")
    print(f"VAL rows: {len(val_df):,}")

    # Save locally
    train_df.to_csv('/tmp/vertex_data/train.csv', index=False)
    val_df.to_csv('/tmp/vertex_data/val.csv', index=False)

    # Upload to GCS
    print("Uploading TRAIN and VAL files to GCS...")
    bucket.blob('training_data/f1_train.csv').upload_from_filename('/tmp/vertex_data/train.csv')
    bucket.blob('training_data/f1_val.csv').upload_from_filename('/tmp/vertex_data/val.csv')

    print("Done!")
    print(f"  gs://{bucket_name}/training_data/f1_train.csv")
    print(f"  gs://{bucket_name}/training_data/f1_val.csv")


if __name__ == "__main__":
    prepare_vertex_data()
