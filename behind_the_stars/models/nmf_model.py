import pandas as pd
import pyarrow.parquet as pq
from ml_logic.preprocessor import preprocessing, vectorizing_tfid,vectorizing_countv,sum_and_sort
from sklearn.feature_extraction.text import TfidfVectorizer
from ml_logic.preprocessor import master_preprocessor
from sklearn.decomposition import NMF
from pandarallel import pandarallel
from models import lda_topics

pandarallel.initialize(progress_bar=True)
