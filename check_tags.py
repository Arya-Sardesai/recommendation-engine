from huggingface_hub import hf_hub_download
try:
    path = hf_hub_download(repo_id='Arya2305/recommendation-engine-data', filename='book_tags.parquet', repo_type='dataset')
    import pandas as pd
    df = pd.read_parquet(path)
    print('Loaded:', len(df), 'rows,', df['book_id'].nunique(), 'books')
    print('Sample book_ids:', df['book_id'].head(5).tolist())
    print('book_id dtype:', df['book_id'].dtype)
except Exception as e:
    print('FAILED:', e)
