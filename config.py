import os

# Index path
INDEX_DIR = "/users/40659055/sharedscratch/pyserini_indexes/msmarco-v2.1-doc-segmented/lucene-inverted.msmarco-v2.1-doc-segmented.20240418.4f9675"

# Output folder
OUTPUT_DIR = "/users/40659055/sharedscratch/trec-rag-pipeline/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "retrieval"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "rerank"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "generation"), exist_ok=True)

# Retrieval settings
TOP_K_RETRIEVE = 100    # retrieve top 100 docs per query
TOP_K_RERANK = 10       # rerank top 100 and keep top 10
TOP_K_GENERATE = 5      # number of docs to feed to generator per query
