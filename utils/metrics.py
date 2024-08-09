from beir.retrieval.evaluation import EvaluateRetrieval
from typing import  Dict

def mrr(qrels: Dict[str, Dict[str, int]], results: Dict[str, Dict[str, float]]):
    MRR = 0.0
    top_hits = {}
    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)  
    
    for query_id in top_hits:
        query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0])    
        for rank, hit in enumerate(top_hits[query_id]):
            if hit[0] in query_relevant_docs:
                MRR += 1.0 / (rank + 1)
                break
    MRR = round(MRR/len(qrels), 5)
    return MRR

def evaluate_retrieval(gt_results, retrieval_results, args):
    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(gt_results, retrieval_results, args.Ks)
    output_results = {
        'mAP': _map,
        'ndcg': ndcg,
        'precision': precision,
        'recall': recall
    }
    output_results["MRR"] = mrr(gt_results, retrieval_results)

    return output_results