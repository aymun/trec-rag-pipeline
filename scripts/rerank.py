import json
import argparse
from sentence_transformers import CrossEncoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="BM25 TREC run file")
    parser.add_argument("--queries", required=True, help="JSONL file of queries")
    parser.add_argument("--output", required=True, help="Output reranked TREC run file")
    parser.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Cross-encoder model")
    parser.add_argument("--topk", type=int, default=100, help="Number of top docs to rerank")
    args = parser.parse_args()

    # Load queries
    queries = {}
    with open(args.queries) as f:
        for line in f:
            q = json.loads(line)
            queries[q["id"]] = q["query"]

    # Load BM25 runs
    runs = {}
    with open(args.input) as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            if qid not in runs:
                runs[qid] = []
            runs[qid].append((docid, float(score)))

    # Initialize Cross-Encoder
    model = CrossEncoder(args.model)

    # Rerank
    reranked_runs = {}
    for qid, docs in runs.items():
        top_docs = docs[:args.topk]
        # CrossEncoder expects list of (query, doc_text) tuples
        # Here we only have doc IDs, replace with actual document text later
        doc_texts = [docid for docid, _ in top_docs]  
        pairs = [(queries[qid], doc_text) for doc_text in doc_texts]
        scores = model.predict(pairs)
        reranked = sorted(zip(doc_texts, scores), key=lambda x: x[1], reverse=True)
        reranked_runs[qid] = reranked

    # Save reranked output
    with open(args.output, "w") as out:
        for qid, docs in reranked_runs.items():
            for rank, (docid, score) in enumerate(docs):
                out.write(f"{qid} Q0 {docid} {rank+1} {score} cross-encoder\n")

if __name__ == "__main__":
    main()

