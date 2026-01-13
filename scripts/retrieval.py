import argparse
import json
import os
from pyserini.search.lucene import LuceneSearcher

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="Path to BM25 index")
    parser.add_argument("--queries", required=True, help="JSONL file of queries")
    parser.add_argument("--output", required=True, help="Output TREC run file")
    parser.add_argument("--topk", type=int, default=100, help="Number of top docs to retrieve")
    args = parser.parse_args()

    # Load queries
    queries = []
    with open(args.queries) as f:
        for line in f:
            q = json.loads(line)
            queries.append(q)

    print("Loading MS MARCO index...")
    searcher = LuceneSearcher(args.index)
    searcher.set_bm25(k1=0.9, b=0.4)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w") as out:
        for q in queries:
            qid = q["query_id"]      # TREC format
            query_text = q["query"]

            hits = searcher.search(query_text, k=args.topk)

            for rank, hit in enumerate(hits):
                out.write(f"{qid} Q0 {hit.docid} {rank+1} {hit.score:.4f} bm25\n")

    print(f"Retrieval done. Results saved to {args.output}")

if __name__ == "__main__":
    main()
