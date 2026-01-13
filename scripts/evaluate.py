# scripts/evaluate.py

import argparse
import pytrec_eval
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", required=True, help="Path to qrels file (TREC format)")
    parser.add_argument("--run", required=True, help="Path to retrieval run file")
    args = parser.parse_args()

    # Load qrels and run
    with open(args.qrels) as f:
        qrels = pytrec_eval.parse_qrel(f)

    with open(args.run) as f:
        run = pytrec_eval.parse_run(f)

    # Set metrics
    metrics_to_compute = {'map', 'ndcg', 'recip_rank'}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics_to_compute)

    # Evaluate
    results = evaluator.evaluate(run)

    # Print per-query metrics
    print("Per-query metrics:")
    for qid, metrics in results.items():
        print(f"{qid}: {metrics}")

    # Compute and print average metrics
    print("\nAverage metrics over all queries:")
    for metric in metrics_to_compute:
        avg = np.mean([metrics[metric] for metrics in results.values()])
        print(f"{metric}: {avg:.4f}")

if __name__ == "__main__":
    main()
