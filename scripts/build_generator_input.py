import argparse
import json
from collections import defaultdict
from pyserini.search.lucene import LuceneSearcher


def load_queries(query_file):
    queries = {}
    with open(query_file) as f:
        for line in f:
            qid, text = line.strip().split("\t", 1)
            queries[qid] = text
    return queries


def load_rerank_results(run_file, topk=10):
    results = defaultdict(list)
    with open(run_file) as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            if len(results[qid]) < topk:
                results[qid].append(docid)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--rerank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    print("Loading index...")
    searcher = LuceneSearcher(args.index)

    print("Loading queries...")
    queries = load_queries(args.queries)

    print("Loading reranked results...")
    reranked = load_rerank_results(args.rerank, args.topk)

    print("Building generator input...")

    with open(args.output, "w") as out:
        for qid, docids in reranked.items():
            contexts = []
            for docid in docids:
                doc = searcher.doc(docid)
                if doc:
                    contexts.append({
                        "docid": docid,
                        "text": doc.raw()
                    })

            record = {
                "query_id": qid,
                "query": queries[qid],
                "contexts": contexts
            }

            out.write(json.dumps(record) + "\n")

    print(f"Generator input saved to: {args.output}")


if __name__ == "__main__":
    main()
