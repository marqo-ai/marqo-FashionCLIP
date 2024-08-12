# Marqo FashionCLIP

This repository is for evaluating Marqo FashionCLIP and Marqo FashionSigLIP against 7 public benchmark datsets. Read more about our models on our [blog](https://www.marqo.ai/blog) and refer to [LEADERBOARD](LEADERBOARD.md) for the performance comparison.

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

## Summarizing Results
To renew [LEADERBOARD.md](LEADERBOARD.md) and summarize results of different models locally, run this command
```bash
python summarize_results.py
```
