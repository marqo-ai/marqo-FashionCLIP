# Marqo FashionCLIP

This repository is for evaluating Marqo FashionCLIP and Marqo FashionSigLIP against 7 public benchmark datasets. Read more about our models on our [blog](https://www.marqo.ai/blog) and refer to [LEADERBOARD](LEADERBOARD.md) for the performance comparison.

## Models
We released our models on HuggingFace: [Marqo-fashionCLIP](https://huggingface.co/Marqo/marqo-fashionCLIP) and [Marqo-fashionSigLIP](https://huggingface.co/Marqo/marqo-fashionSigLIP). You can load the models with open_clip by

```python
import open_clip
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionCLIP')
tokenizer = open_clip.get_tokenizer('hf-hub:Marqo/marqo-fashionCLIP')
```
and
```python
import open_clip
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionSigLIP')
tokenizer = open_clip.get_tokenizer('hf-hub:Marqo/marqo-fashionSigLIP')
```

Scripts to evalute other models including [FashionCLIP 2.0](https://github.com/patrickjohncyh/fashion-clip) and [OpenFashionCLIP](https://github.com/aimagelab/open-fashion-clip) can be found in [scripts](scripts) directory.

## Installation
Install PyTorch first and run 
```bash
pip -r requirements.txt
```

## Quick Start

To evaluate Marqo FashionCLIP, run this command
```bash
python eval.py \
        --dataset-config ./configs/${DATASET}.json \
        --model-name Marqo/marqo-fashionCLIP \
        --run-name Marqo-FashionCLIP
```
- `DATASET` can be one of ['deepfashion_inshop', 'deepfashion_multimodal', 'fashion200k', 'KAGL', 'atlas', 'polyvore' 'iMaterialist']

To evaluate Marqo FashionSigLIP, run this command
```bash
python eval.py \
        --dataset-config ./configs/${DATASET}.json \
        --model-name Marqo/marqo-fashionSigLIP \
        --run-name Marqo-FashionSigLIP
```
- `DATASET` can be one of ['deepfashion_inshop', 'deepfashion_multimodal', 'fashion200k', 'KAGL', 'atlas', 'polyvore' 'iMaterialist']


## Datasets
We collected 7 public multimodal fashion datasets and uploaded to HuggingFace: [Atlas](https://huggingface.co/datasets/Marqo/atlas), [DeepFashion (In-shop)](https://huggingface.co/datasets/Marqo/deepfashion-inshop), [DeepFashion (Multimodal)](https://huggingface.co/datasets/Marqo/deepfashion-multimodal), [Fashion200k](https://huggingface.co/datasets/Marqo/fashion200k), [iMaterialist](https://huggingface.co/datasets/Marqo/iMaterialist), [KAGL](https://huggingface.co/datasets/Marqo/KAGL), and [Polyvore](https://huggingface.co/datasets/Marqo/polyvore). Each dataset has different metadata available. Thus, tasks for each dataset are stored as json files in [scripts](scripts) directory. Refer to our [blog](https://www.marqo.ai/blog) for more information about each dataset.

## Benchmark Results
We averaged performance of three common tasks across the datasets: text-to-image, category-to-product, and sub-category-to-product. As shown belowe, Marqo-FashionCLIP and Marqo-FashionSigLIP outperform both pretrained OpenCLIP models and the state-of-the-art fashion CLIP models. Refer to [LEADERBOARD](LEADERBOARD.md) for more comprehensive results.

**Text-To-Image (Averaged across 6 datasets)**
| Model                      | AvgRecall   | Recall@1   | Recall@10   | MRR       |
|----------------------------|-------------|------------|-------------|-----------|
| FashionCLIP2.0                | 0.163       | 0.077      | 0.249       | 0.165     |
| Marqo-FashionCLIP          | 0.192       | 0.094      | 0.290       | 0.200     |
| Marqo-FashionSigLIP        | **0.231**   | **0.121**  | **0.340**   | **0.239** |
| OpenFashionCLIP            | 0.132       | 0.060      | 0.204       | 0.135     |
| ViT-B-16-laion2b_s34b_b88k | 0.174       | 0.088      | 0.261       | 0.180     |
| ViT-B-16-SigLIP-webli      | 0.212       | 0.111      | 0.314       | 0.214     |

**Category-To-Product (Averaged across 5 datasets)**
| Model                      | AvgP      | P@1       | P@10      | MRR       |
|----------------------------|-----------|-----------|-----------|-----------|
| FashionCLIP2.0                | 0.684     | 0.681     | 0.686     | 0.741     |
| Marqo-FashionCLIP          | 0.705     | 0.734     | 0.676     | 0.776     |
| Marqo-FashionSigLIP        | **0.737** | **0.758** | **0.716** | **0.812** |
| OpenFashionCLIP            | 0.646     | 0.653     | 0.639     | 0.720     |
| ViT-B-16-laion2b_s34b_b88k | 0.662     | 0.673     | 0.652     | 0.743     |
| ViT-B-16-SigLIP-webli      | 0.688     | 0.690     | 0.685     | 0.751     |

**Sub-Category-To-Product (Averaged across 4 datasets)**
| Model                      | AvgP      | P@1       | P@10      | MRR       |
|----------------------------|-----------|-----------|-----------|-----------|
| FashionCLIP2.0                | 0.657     | 0.676     | 0.638     | 0.733     |
| Marqo-FashionCLIP          | 0.707     | 0.747     | 0.667     | 0.772     |
| Marqo-FashionSigLIP        | **0.725** | **0.767** | **0.683** | **0.811** |
| OpenFashionCLIP            | 0.598     | 0.619     | 0.578     | 0.689     |
| ViT-B-16-laion2b_s34b_b88k | 0.638     | 0.651     | 0.624     | 0.712     |
| ViT-B-16-SigLIP-webli      | 0.643     | 0.643     | 0.643     | 0.726     |

## Summarizing Results
To renew [LEADERBOARD.md](LEADERBOARD.md) and summarize results of different models locally, run this command
```bash
python summarize_results.py
```
