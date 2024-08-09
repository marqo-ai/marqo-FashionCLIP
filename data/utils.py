from datasets import load_dataset
import logging
import os

class Transform(object):
    def __init__(self, tokenizer, preprocess, doc_text_cols):
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.doc_text_cols = doc_text_cols
    def __call__(self, batch):
        if 'image' in batch:
            batch['image'] = [self.preprocess(img) for img in batch['image']]
            
        if self.doc_text_cols:
            for col in self.doc_text_cols:
                batch[col] = [self.tokenizer(text)[0] for text in batch[col]]
        return batch

def get_dataset(args, tokenizer, preprocess):
    logging.info('Loading dataset from huggingface.')
    doc_dataset = load_dataset(args.dataset_config["hf_dataset"], num_proc=os.cpu_count(), cache_dir=args.cache_dir)['data']
    
    doc_text_cols, query_text_cols = set(), set()
    for task in args.dataset_config["tasks"]:
        # Document columns
        for doc_col in task["doc_col"]:
            if doc_col != 'image':
                doc_text_cols.add(doc_col)
        # Query columns
        for query_col in task["query_col"]:
            query_text_cols.add(query_col)
    item_ID = [str(id) for id in doc_dataset.data['item_ID'].to_pylist()]
    doc_dataset = doc_dataset.remove_columns([col for col in doc_dataset.column_names if col!='image' and (col not in doc_text_cols)])
            
    # Apply transform
    transform = Transform(tokenizer, preprocess, list(doc_text_cols))
    doc_dataset.set_transform(transform)
        
    return doc_dataset, item_ID