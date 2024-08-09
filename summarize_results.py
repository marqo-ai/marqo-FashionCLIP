import argparse
import pandas as pd
import os
import json
from glob import glob
import numpy as np
import datetime
import sys

parser = argparse.ArgumentParser()

parser.add_argument("--result-dir", type=str, default="./results", help='Result directory.')
parser.add_argument('--dataset-config-dir', default='./configs', help='Dataset config directory.')
parser.add_argument("--summary-dir", type=str, default="./summary", help='Summary directory.')
parser.add_argument('--t2i-metrics', 
                    default=["Recall@1", "Recall@10", "MRR"], 
                    nargs='+', 
                    help='Metrics to dispaly for the text-to-image task.')
parser.add_argument('--c2p-metrics', 
                    default=["P@1", "P@10", "MRR"], 
                    nargs='+', 
                    help='Metrics to dispaly for category-to-product task.')
parser.add_argument('--tasks-avg-results', 
                    default=["text-to-image","category-to-product","sub-category-to-product"], 
                    nargs='+', 
                    help='Tasks to average across datasets.')

args = parser.parse_args()

if __name__ == "__main__":
    if not os.path.exists(args.summary_dir):
        os.makedirs(args.summary_dir)
    display_metrics = {'t2i': ["AvgRecall"]+args.t2i_metrics, 'c2p': ["AvgP"]+args.c2p_metrics}

    all_md = ""
    avg_results = {k: {'Model': []} for k in args.tasks_avg_results}
    eval_datasets = []
    _dec_format = '{:.3f}'
    for dataset in sorted(os.listdir(args.result_dir), key=str.lower):
        model_info = {'Model': []}
        
        dataset_result_dir = os.path.join(args.result_dir, dataset) # ex) results/deepfashion_inshop
        with open(os.path.join(args.dataset_config_dir, f'{dataset}.json'), 'r') as f:
            dataset_config = json.load(f)
        dataset_name = dataset_config['name']
        all_md += f"## {dataset_name}\n"
        eval_datasets.append(dataset_name)

        task_results = {}
        for model in sorted(os.listdir(dataset_result_dir), key=str.lower): 
            dataset_model_result_dir = os.path.join(dataset_result_dir, model) # ex) results/deepfashion_inshop/ViT-B-32-laion2b_e16
            model_info['Model'].append(model)
            
            for task in dataset_config['tasks']:
                task_name = task['name']
                metric_type = 't2i'if task_name.lower() == 'text-to-image' else 'c2p'
                _metric = display_metrics[metric_type]
                if task_name not in task_results:
                    task_results[task_name] = {m: [] for m in _metric}

                if task_name in avg_results:
                    for m in _metric:
                        if m not in avg_results[task_name]:
                            avg_results[task_name][m] = []
                    if model in avg_results[task_name]['Model']:
                        ind = np.where((np.asarray(avg_results[task_name]['Model'])==model))[0][0]
                    else:
                        avg_results[task_name]['Model'].append(model)
                        ind = len(avg_results[task_name]['Model'])-1
                        for m in _metric:
                            if 'Avg' not in m:
                                avg_results[task_name][m].append([])

                task_result_dir = os.path.join(dataset_model_result_dir, task_name)
                result_json = glob(os.path.join(task_result_dir, 'result*.json'))
                assert len(result_json)==1, f'{len(result_json)} result files are found in {task_result_dir}.'
                with open(result_json[0], 'r') as f:
                    result = json.load(f)
                for _metric, metric_result in result.items():
                    if isinstance(metric_result, dict):
                        for _metric, _value in metric_result.items():
                            if _metric in task_results[task_name]:
                                task_results[task_name][_metric].append(float(_value))
                                if task_name in avg_results:
                                    avg_results[task_name][_metric][ind].append(float(_value))
                    else:
                        if _metric in task_results[task_name]:
                            task_results[task_name][_metric].append(float(metric_result))
                            if task_name in avg_results:
                                avg_results[task_name][_metric][ind].append(float(metric_result))
        # Average metrics
        for task_name in task_results.keys():
            metrics_to_avg = args.t2i_metrics if task_name.lower() == 'text-to-image' or dataset_config['name'] == 'Shopify' else args.c2p_metrics
            agg_metrics = {}
            for m in metrics_to_avg:
                if '@' in m:
                    if m.split('@')[0] not in agg_metrics:
                        agg_metrics[m.split('@')[0]] = [m]
                    else:
                        agg_metrics[m.split('@')[0]].append(m)
            for m,v in agg_metrics.items():
                task_results[task_name][f"Avg{m}"] = np.mean([task_results[task_name][_v] for _v in v], axis=0).tolist()
                if task_name in avg_results:
                    avg_results[task_name][f"Avg{m}"].append(np.mean([task_results[task_name][_v] for _v in v], axis=0).tolist())

        # Make the best result bold font
        for task_name in task_results.keys():
            for m in task_results[task_name].keys():
                if not task_results[task_name][m] or not isinstance(task_results[task_name][m][0], float):
                    continue
                best_ind = np.argmax([val for val in task_results[task_name][m]])
                best_val = _dec_format.format(task_results[task_name][m][best_ind])
                for i in range(len(task_results[task_name][m])):
                    _val = _dec_format.format(task_results[task_name][m][i])
                    if _val == best_val:
                        task_results[task_name][m][i] = '**'+_val+'**'
                    else:
                        task_results[task_name][m][i] = _val
        
        # Convert to markdown by pd.DataFrame
        if not os.path.exists(os.path.join(args.summary_dir, dataset_name)):
            os.makedirs(os.path.join(args.summary_dir, dataset_name))
        for task_name, result in task_results.items():
            _tmp_result = model_info.copy()
            _tmp_result.update({k: v for k,v in result.items() if len(v)>0})
            all_md += f'### {task_name.title()}\n'
            all_md += pd.DataFrame(_tmp_result).to_markdown(index=False, tablefmt="github")
            all_md += '\n'
            df = pd.DataFrame({k: [_v.replace('**','') for _v in v] for k,v in _tmp_result.items()})
            df.to_csv(os.path.join(args.summary_dir, dataset_name, f'{task_name.title()}.csv'), index=False)

    md_prefix = "# Leaderboard\n"
    md_prefix += f"- UPDATED: {str(datetime.datetime.now())[:10].replace(' ', '-')}\n"
    md_prefix += f'- Number of evaluation datasets: {len(eval_datasets)}\n'
    md_prefix += f"- Datasets: {', '.join(['[{}](#{})'.format(d, d.lower().replace(' ','-'))for d in eval_datasets])}\n"
    
    md_prefix += '## Average Results\n'
    for task_name, task_result in avg_results.items():
        _n_datasets = set()
        for k, v in task_result.items():
            if 'Avg' in k:
                avg_results[task_name][k] = np.mean(v, axis=0).tolist()
            elif isinstance(v[0], list):
                avg_results[task_name][k] = [np.mean(_v) for _v in v]
                for _v in v:
                    _n_datasets.add(len(_v))
        assert len(_n_datasets)==1
        _n_datasets = list(_n_datasets)[0]
        for m in avg_results[task_name].keys():
            if not avg_results[task_name][m] or not isinstance(avg_results[task_name][m][0], float):
                continue
            best_ind = np.argmax([val for val in avg_results[task_name][m]])
            best_val = _dec_format.format(avg_results[task_name][m][best_ind])
            for i in range(len(avg_results[task_name][m])):
                _val = _dec_format.format(avg_results[task_name][m][i])
                if _val == best_val:
                    avg_results[task_name][m][i] = '**'+_val+'**'
                else:
                    avg_results[task_name][m][i] = _val
        md_prefix += f'### {task_name.title()} (Averaged across {_n_datasets} datasets)\n'
        md_prefix += pd.DataFrame({k: v for k,v in avg_results[task_name].items() if ''.join(v)!=''}).to_markdown(index=False, tablefmt="github")
        md_prefix += '\n'
        
    with open('LEADERBOARD.md', 'w') as f:
        f.write(md_prefix+all_md)

