import numpy as np
v = np.load("data/processed/embeddings_bgem3.npy")           # (432715, 1024) float32
np.save("data/processed/embeddings_bgem3_fp16.npy", v.astype(np.float16))
# then upload embeddings_bgem3_fp16.npy to the dataset repo via your upload_data.py