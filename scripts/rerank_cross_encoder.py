import argparse
from sentence_transformers import CrossEncoder
import os

def load_retrieval_results(filepath):
    """Load retrieval results from BM25 run file."""
    results = {}
    with open(filepath, "r") as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            if qid not in results:
                results[qid] = []
            results[qid].append(docid)
    return results

def rerank(query_file, retrieval_results_file, output_file, model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"):
    # Load queries
    queries = {}
    with open(query_file, "r") as f:
        for line in f:
            qid, query_text = line.strip().split("\t", 1)
            queries[qid] = query_text

    # Load retrieval results
    retrieval_results = load_retrieval_results(retrieval_results_file)

    # Load cross-encoder model
    print("Loading cross-encoder model...")
    model = CrossEncoder(model_name)

    # Open output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as out:
        for qid, docs in retrieval_results.items():
            query_text = queries[qid]
            # Prepare pairs for cross-encoder
            pairs = [[query_text, docid] for docid in docs]  # Here, docid is just the ID; if you have actual doc text, replace docid with the text
            scores = model.predict(pairs)
            # Sort by score descending
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            for rank, (docid, score) in enumerate(ranked, start=1):
                out.write(f"{qid} Q0 {docid} {rank} {score:.8f} cross-encoder\n")
    print(f"Reranking done. Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, help="Test queries file (txt with tab-separated ID and text)")
    parser.add_argument("--retrieval", required=True, help="BM25 retrieval results file")
    parser.add_argument("--output", required=True, help="Output reranked run file")
    parser.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-12-v2", help="Cross-encoder model name")
    args = parser.parse_args()

    rerank(args.queries, args.retrieval, args.output, args.model)
