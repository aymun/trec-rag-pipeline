# scripts/evaluate.py
import pytrec_eval
import json

# Paths
qrels_file = "data/trec2024_qrels.txt"
run_file = "outputs/retrieval/retrieval_results.txt"
output_file = "outputs/retrieval/evaluation_summary.txt"

# Load qrels (ensure relevance labels are integers)
qrels = {}
with open(qrels_file) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][docid] = rel

# Load run file
run = {}
with open(run_file) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        qid, _, docid, rank, score, _ = parts
        if qid not in run:
            run[qid] = {}
        run[qid][docid] = float(score)

# Evaluate
evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg', 'recip_rank'})
results = evaluator.evaluate(run)

# Compute averages
avg_map = sum([r['map'] for r in results.values()]) / len(results)
avg_ndcg = sum([r['ndcg'] for r in results.values()]) / len(results)
avg_rr = sum([r['recip_rank'] for r in results.values()]) / len(results)

summary = f"""
Average metrics over all queries:
map: {avg_map:.4f}
ndcg: {avg_ndcg:.4f}
recip_rank: {avg_rr:.4f}
"""

# Save to file
with open(output_file, "w") as f:
    f.write(summary)

# Print to console
print(summary)
