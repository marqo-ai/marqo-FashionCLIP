from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from collections import defaultdict

def get_embeddings(model, doc_dataset, args):
    embeddings = defaultdict(list)
    eval_dataloader = DataLoader(doc_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for sample_type, data in batch.items():
                if sample_type == 'image':
                    embeddings[sample_type].append(model.encode_image(data.to(args.device), normalize=True).to('cpu'))
                else:
                    embeddings[sample_type].append(model.encode_text(data.to(args.device), normalize=True).to('cpu'))
    for k, v in embeddings.items():
        embeddings[k] = torch.cat(v, dim=0)
    return embeddings

def run_retrieval(test_queries, docID, embeddings, tokenizer, model, k, args):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    results = dict()
    assert len(docID) == embeddings.shape[0]

    for query in tqdm(test_queries, total=len(test_queries)):
        if model is not None:
            text = tokenizer([args.query_prefix + query]).to(args.device)

            with torch.no_grad(), torch.amp.autocast('cuda'):
                text_features = model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = cos(text_features.to(args.device), embeddings)

            top_scores, top_inds = torch.topk(similarity, k)
            top_scores = list(top_scores.cpu().numpy())
            top_inds = list(top_inds.cpu().numpy())
            results[query] = {str(docID[idx]): float(_s) for idx, _s in zip(top_inds, top_scores)}
    return results