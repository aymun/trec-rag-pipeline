# scripts/evaluate_pipeline.py
import json
import pytrec_eval
import argparse

def load_qrels(qrels_file):
    qrels = {}
    with open(qrels_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, docid, rel = parts[:4]
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][docid] = int(rel)
    return qrels

def load_run(run_file):
    run = {}
    with open(run_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, _, docid, rank, score, _ = parts[:6]
                if qid not in run:
                    run[qid] = {}
                run[qid][docid] = float(score)
    return run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", required=True, help="Path to qrels.txt")
    parser.add_argument("--run", required=True, help="Path to run file")
    parser.add_argument("--output", required=True, help="Output file for per-query metrics")
    args = parser.parse_args()

    qrels = load_qrels(args.qrels)
    run = load_run(args.run)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg', 'recip_rank'})
    results = evaluator.evaluate(run)

    # Save per-query results
    with open(args.output, "w") as f:
        for qid, metrics in results.items():
            f.write(f"{qid} {json.dumps(metrics)}\n")

    # Compute averages
    avg_map = sum([m['map'] for m in results.values()]) / len(results)
    avg_ndcg = sum([m['ndcg'] for m in results.values()]) / len(results)
    avg_rr = sum([m['recip_rank'] for m in results.values()]) / len(results)

    print("\nAverage metrics over all queries:")
    print(f"map: {avg_map:.4f}")
    print(f"ndcg: {avg_ndcg:.4f}")
    print(f"recip_rank: {avg_rr:.4f}\n")

if __name__ == "__main__":
    main()
