import pandas as pd
import os
import pyarrow.parquet as pq
import pyarrow as pa
import gcsfs
from google.cloud import storage
from ml_logic. params import BUCKET_NAME, DATA_FILENAME, LOCAL_DATA_PATH, RANDOM_STATE

def get_data(source="cloud"):
    """This function loads the data.
    Parameter source may be either "cloud" or "local".
    It requires to get pyarrow, gcsfs and pandas installed from requirements.
    gcsfs is important in order to load when source is "cloud".
    Returns pandas dataframe.
    """
    local_path = os.path.join(LOCAL_DATA_PATH, DATA_FILENAME)

    if os.path.exists(local_path):
        print(f"Loading from local: {local_path}")
        return pd.read_parquet(local_path)

    elif source == "cloud":

        os.makedirs(LOCAL_DATA_PATH, exist_ok=True)

        try:
            client = storage.Client()
            bucket = client.bucket(BUCKET_NAME)
            blob = bucket.blob(DATA_FILENAME)

            blob.download_to_filename(local_path)
            print("Download success.")

            return pd.read_parquet(local_path)

        except Exception as e:
            print(f"GC Connection Error: {e}")
            return None

    else:
        print(f"File doesn't exist in {local_path}")
        return None


def get_balanced_data(df, n_total=None):
    """ This function takes a dataframe and returns a 50/50 balanced sample
    according to "is_open" feature. n_total can be used to determine the size of
    the sample
    """
    df_0 = df[df["is_open"] == 0]
    df_1 = df[df["is_open"] == 1]

    min_size = min(len(df_0), len(df_1))

    if n_total is not None:
        size_per_class = min(n_total // 2, min_size)
    else:
        size_per_class = min_size

    df_0_sampled = df_0.sample(n= size_per_class, random_state = RANDOM_STATE)
    df_1_sampled = df_1.sample(n= size_per_class, random_state = RANDOM_STATE)

    balanced_df = pd.concat([df_0_sampled, df_1_sampled])


def get_initial_slice(n_rows=10000):
    """
    Downloads 10000 lines in order to be able to work
    """
    path = f"gs://{BUCKET_NAME}/{DATA_FILENAME}"
    print(f"Accesing {path} for a fast download.")

    fs = gcsfs.GCSFileSystem(token="google_default")

    with fs.open(path) as f:
        parquet_file = pq.ParquetFile(f)

        batches = parquet_file.iter_batches(batch_size=n_rows)

        first_batch = next(batches)

        df_slice = pa.Table.from_batches([first_batch]).to_pandas()

    print(f"✅ Partial download completed: {len(df_slice)} rows")
    return df_slice
