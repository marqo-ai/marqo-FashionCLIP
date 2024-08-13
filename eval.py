import torch
import argparse
import json
import os
from utils import get_embeddings, run_retrieval, evaluate_retrieval, setup_logging
from data.utils import get_dataset
from models import load_model
import torch.nn.functional as F
import logging

parser = argparse.ArgumentParser()

# Args for datasets
parser.add_argument("--data-dir", type=str, default="./data/", help='Data directory.')
parser.add_argument('--dataset-config', default='./configs/deepfashion_inshop.json', help='Dataset config file.')
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument("--num-workers", type=int, default=4)
# Args for models
parser.add_argument('--model-name', type=str, default='ViT-B-16', help='Model name.')
parser.add_argument('--run-name', type=str, default='ViT-B-16_laion2b_s34b_b88k', help='Run name.')
parser.add_argument("--pretrained", type=str, default='laion2b_s34b_b88k', help='Pretrained name.')
parser.add_argument('--cache-dir', default=".cache", help='Cache directory for models and datasets.')
parser.add_argument('--device', default='cuda', help='Device to use for inference.')
parser.add_argument("--query-prefix", type=str, default='', help="Query prefix if required (ex. 'description: ')")
# Args for evaluations
parser.add_argument('--Ks', default=[1, 10], nargs='+', help='Ks for metrics.')
parser.add_argument("--overwrite-embeddings", action="store_true", default=False)
parser.add_argument("--overwrite-retrieval", action="store_true", default=False)
parser.add_argument("--output-dir", type=str, default='./results')

args = parser.parse_args()

if __name__ == "__main__":
    setup_logging()
    # Output directory settings
    args.output_dir = os.path.join(args.output_dir, os.path.basename(args.dataset_config).replace('.json',''), args.run_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        logging.warning(f'Output directory {args.output_dir} exists. Ignore this if it is expected.')
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    args.embeddings_path = os.path.join(args.output_dir, "embeddings.pt")
    
    # Read dataset config file
    with open(args.dataset_config, 'r') as file:
        args.dataset_config = json.load(file)
    logging.info("Dataset: " + args.dataset_config["name"])

    # Load model
    model, preprocess, tokenizer = load_model(args)

    # Load documenets and generate embeddings
    model = model.to(args.device)

    doc_dataset, item_ID = get_dataset(args, tokenizer, preprocess)
    logging.info(f"Number of document rows: {len(doc_dataset):,}")

    if not os.path.isfile(args.embeddings_path) or args.overwrite_embeddings:
        logging.info("Getting embeddings of documents")
        embeddings = get_embeddings(model, doc_dataset, args)
        torch.save(embeddings, args.embeddings_path)
    else:
        logging.info("Loading embeddings of documents")
        embeddings = torch.load(args.embeddings_path)

    # Run tasks
    for task in args.dataset_config["tasks"]:
        task_dir = os.path.join(args.output_dir, task['name'])
        if not os.path.exists(task_dir):
            os.makedirs(task_dir, exist_ok=True)
        logging.info(f'Task: {json.dumps(task, indent=4)}')

        for query_col in task["query_col"]:
            gt_dir = os.path.join(args.data_dir, args.dataset_config["name"], 'gt_query_doc')
            gt_results_path = os.path.join(gt_dir, f"ground_truth_{query_col}-{'+'.join(task['doc_col'])}.json")
            assert os.path.exists(gt_results_path)

            # Ground-truth query-doc
            logging.info("Loading ground truth")
            with open(gt_results_path, "r") as f:
                gt_results = json.load(f)
                test_queries = list(gt_results.keys()) # randomly sampled queries (up to 2000)
            
            # Running retrieval
            retrieval_path = os.path.join(task_dir, f"retrieved_{query_col}-{'+'.join(task['doc_col'])}.json")
            if os.path.exists(retrieval_path) and not args.overwrite_retrieval:
                logging.info("Loading retrieval")
                with open(retrieval_path, "r") as f:
                    retrieval_results = json.load(f)
            else:
                logging.info("Running retrieval")
                if len(task['doc_col'])==1:
                    doc_embeddings = embeddings[task['doc_col'][0]].to(args.device)
                else:
                    assert ('doc_weights' in task and len(task['doc_weights'])==len(task['doc_col'])), \
                        "Must provide the same number of weights for multi-field documents as the number of multi-fields."
                    doc_embeddings = F.normalize(torch.stack([w*embeddings[c] for c, w in zip(task['doc_col'], task['doc_weights'])], dim=1).sum(1), dim=-1).to(args.device)
                retrieval_results = run_retrieval(test_queries, item_ID, doc_embeddings, tokenizer, model, max(args.Ks), args)
                with open(retrieval_path, "w") as f:
                    json.dump(retrieval_results, f, indent=4)

            # Evaluation Starts
            logging.info("Evaluation Starts")
            output_results = evaluate_retrieval(gt_results, retrieval_results, args)
            output_json = os.path.join(task_dir, f"result_{query_col}-{'+'.join(task['doc_col'])}.json")
            output_json_dict = json.dumps(output_results, indent=4)
            logging.info(output_json_dict)
            with open(output_json, 'w') as f:
                f.write(output_json_dict)