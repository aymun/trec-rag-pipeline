# scripts/retrieval.py
import argparse
from pyserini.search.lucene import LuceneSearcher

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="Path to BM25 index")
    parser.add_argument("--queries", required=True, help="TXT file with queries (tab-separated: qid<TAB>query)")
    parser.add_argument("--output", required=True, help="Output TREC run file")
    parser.add_argument("--topk", type=int, default=100, help="Number of top docs to retrieve")
    args = parser.parse_args()

    # Load queries from TXT file
    queries = []
    with open(args.queries, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, query_text = line.split("\t", 1)  # split at first tab
            queries.append((qid, query_text))

    # Initialize searcher
    print("Loading MS MARCO index...")
    searcher = LuceneSearcher(args.index)
    print("Index loaded successfully!")

    # Open output file
    with open(args.output, "w") as out:
        for qid, query_text in queries:
            hits = searcher.search(query_text, k=args.topk)
            for rank, hit in enumerate(hits, start=1):
                # TREC-compliant format
                out.write(f"{qid} Q0 {hit.docid} {rank} {hit.score:.8f} bm25\n")

    print(f"Retrieval done. Results saved to {args.output}")

if __name__ == "__main__":
    main()
