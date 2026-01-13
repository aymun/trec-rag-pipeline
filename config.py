# config.py
import os

# =========================
# Paths
# =========================

# Location of the MS MARCO index (prebuilt in shared scratch)
INDEX_DIR = "/users/40659055/sharedscratch/pyserini_indexes/msmarco-v2.1-doc-segmented"

# Output folder for pipeline results
OUTPUT_DIR = "/users/40659055/sharedscratch/trec_rag_outputs"

# Stage-specific subfolders
RETRIEVAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "retrieval")
RERANK_OUTPUT_DIR   = os.path.join(OUTPUT_DIR, "rerank")
GEN_OUTPUT_DIR      = os.path.join(OUTPUT_DIR, "generation")

# Create folders if they don't exist
for d in [RETRIEVAL_OUTPUT_DIR, RERANK_OUTPUT_DIR, GEN_OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# =========================
# Retrieval settings
# =========================

# Number of documents retrieved per query from Lucene/BM25 index
TOP_K_RETRIEVE = 100

# =========================
# Reranking settings
# =========================

# Number of documents selected after reranking
TOP_K_RERANK = 10

# Cross-encoder / sentence-transformer model for reranking
# e.g., 'cross-encoder/ms-marco-MiniLM-L-6-v2'
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# =========================
# Generation settings
# =========================

# Number of top reranked documents to feed into the generator
TOP_K_GENERATE = 5

# Generative model for RAG
# e.g., 'facebook/rag-token-base' or any LLM
GEN_MODEL_NAME = "facebook/rag-token-base"

# Maximum length of generated answer
GEN_MAX_LENGTH = 256

# Device for generation ('cuda' for GPU, 'cpu' for CPU)
GEN_DEVICE = "cuda"

# =========================
# Other settings
# =========================

# Batch size for generation
GEN_BATCH_SIZE = 4

# Logging verbosity
VERBOSE = True
