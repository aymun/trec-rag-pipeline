# run_pipeline.py
import os
from scripts.retrieval import run_retrieval
from scripts.rerank import run_reranking
from scripts.generate import run_generation
from config import OUTPUT_DIR, TOP_K

def main():
    """
    Full RAG pipeline:
    1. Retrieve top documents for queries
    2. Rerank retrieved documents
    3. Generate responses using top reranked documents
    """

    print("===== STARTING RAG PIPELINE =====")

    # 1️⃣ Retrieval
    print("\n[STEP 1] Running retrieval...")
    retrieval_results = run_retrieval(top_k=TOP_K)
    retrieval_output_file = os.path.join(OUTPUT_DIR, "retrieval", "retrieval_results.json")
    print(f"Saving retrieval results to: {retrieval_output_file}")
    
    # 2️⃣ Reranking
    print("\n[STEP 2] Running reranking...")
    rerank_results = run_reranking(
        retrieval_results=retrieval_results,
        top_k=10  # after reranking, keep top 10 per query
    )
    rerank_output_file = os.path.join(OUTPUT_DIR, "rerank", "rerank_results.json")
    print(f"Saving rerank results to: {rerank_output_file}")

    # 3️⃣ Generation
    print("\n[STEP 3] Running generation...")
    generate_results = run_generation(rerank_results=rerank_results)
    generation_output_file = os.path.join(OUTPUT_DIR, "generation", "generation_results.json")
    print(f"Saving generation results to: {generation_output_file}")

    print("\n===== RAG PIPELINE COMPLETED =====")
    print(f"Retrieval results: {retrieval_output_file}")
    print(f"Rerank results: {rerank_output_file}")
    print(f"Generation results: {generation_output_file}")

if __name__ == "__main__":
    main()
