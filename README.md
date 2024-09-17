# Marqo-FashionCLIP

This repository is designed to evaluate Marqo-FashionCLIP and Marqo-FashionSigLIP across seven public benchmark datasets. Read more about the models on our [blog](https://www.marqo.ai/blog/search-model-for-fashion).

## Benchmark Results
We averaged the performance of three common tasks across the datasets: text-to-image, category-to-product, and sub-category-to-product. As demonstrated below, Marqo-FashionCLIP and Marqo-FashionSigLIP outperform both pretrained OpenCLIP models and the state-of-the-art fashion CLIP models. For a more comprehensive performance comparison, refer to the [LEADERBOARD](LEADERBOARD.md).

**Text-To-Image (Averaged across 6 datasets)**
| Model                      | AvgRecall   | Recall@1   | Recall@10   | MRR       |
|----------------------------|-------------|------------|-------------|-----------|
| Marqo-FashionSigLIP        | **0.231**   | **0.121**  | **0.340**   | **0.239** |
| Marqo-FashionCLIP          | 0.192       | 0.094      | 0.290       | 0.200     |
| FashionCLIP2.0             | 0.163       | 0.077      | 0.249       | 0.165     |
| OpenFashionCLIP            | 0.132       | 0.060      | 0.204       | 0.135     |
| ViT-B-16-laion2b_s34b_b88k | 0.174       | 0.088      | 0.261       | 0.180     |
| ViT-B-16-SigLIP-webli      | 0.212       | 0.111      | 0.314       | 0.214     |

**Category-To-Product (Averaged across 5 datasets)**
| Model                      | AvgP      | P@1       | P@10      | MRR       |
|----------------------------|-----------|-----------|-----------|-----------|
| Marqo-FashionSigLIP        | **0.737** | **0.758** | **0.716** | **0.812** |
| Marqo-FashionCLIP          | 0.705     | 0.734     | 0.676     | 0.776     |
| FashionCLIP2.0             | 0.684     | 0.681     | 0.686     | 0.741     |
| OpenFashionCLIP            | 0.646     | 0.653     | 0.639     | 0.720     |
| ViT-B-16-laion2b_s34b_b88k | 0.662     | 0.673     | 0.652     | 0.743     |
| ViT-B-16-SigLIP-webli      | 0.688     | 0.690     | 0.685     | 0.751     |

**Sub-Category-To-Product (Averaged across 4 datasets)**
| Model                      | AvgP      | P@1       | P@10      | MRR       |
|----------------------------|-----------|-----------|-----------|-----------|
| Marqo-FashionSigLIP        | **0.725** | **0.767** | **0.683** | **0.811** |
| Marqo-FashionCLIP          | 0.707     | 0.747     | 0.667     | 0.772     |
| FashionCLIP2.0             | 0.657     | 0.676     | 0.638     | 0.733     |
| OpenFashionCLIP            | 0.598     | 0.619     | 0.578     | 0.689     |
| ViT-B-16-laion2b_s34b_b88k | 0.638     | 0.651     | 0.624     | 0.712     |
| ViT-B-16-SigLIP-webli      | 0.643     | 0.643     | 0.643     | 0.726     |

## Models
### Hugging Face
We released our models on HuggingFace: [Marqo-FashionCLIP](https://huggingface.co/Marqo/marqo-fashionCLIP) and [Marqo-FashionSigLIP](https://huggingface.co/Marqo/marqo-fashionSigLIP). We also have a Hugging Face Space Demo of our models in action: [Classification with Marqo-FashionSigLIP](https://huggingface.co/spaces/Marqo/Marqo-FashionSigLIP-Classification).

You can load the models with `transformers` by

```python
from transformers import AutoModel, AutoProcessor
model = AutoModel.from_pretrained('Marqo/marqo-fashionCLIP', trust_remote_code=True)
processor = AutoProcessor.from_pretrained('Marqo/marqo-fashionCLIP', trust_remote_code=True)
```
and
```python
from transformers import AutoModel, AutoProcessor
model = AutoModel.from_pretrained('Marqo/marqo-fashionSigLIP', trust_remote_code=True)
processor = AutoProcessor.from_pretrained('Marqo/marqo-fashionSigLIP', trust_remote_code=True)
```
Then,
```python
import torch
from PIL import Image

image = [Image.open("docs/fashion-hippo.png")]
text = ["a hat", "a t-shirt", "shoes"]
processed = processor(text=text, images=image, padding='max_length', return_tensors="pt")

with torch.no_grad():
    image_features = model.get_image_features(processed['pixel_values'], normalize=True)
    text_features = model.get_text_features(processed['input_ids'], normalize=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
```

We released this [article](https://www.marqo.ai/blog/ecommerce-image-classification-with-marqo-fashionclip) illustrating a simple ecommerce search with a fashion dataset if you want to see the model in action.

### OpenCLIP
You can load the models with `open_clip` by

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
Then,
```python
import torch
from PIL import Image

image = preprocess_val(Image.open("docs/fashion-hippo.png")).unsqueeze(0)
text = tokenizer(["a hat", "a t-shirt", "shoes"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image, normalize=True)
    text_features = model.encode_text(text, normalize=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
```

### Marqo

To deploy on Marqo Cloud (recommended):
1. [Sign Up](https://cloud.marqo.ai/) to [Marqo Cloud](https://cloud.marqo.ai/).

2. Install Marqo and the Marqo python client:
```bash
pip install marqo
```

3. Create and index:

```python
import marqo

settings = {
    "type": "unstructured",
    "model": "marqo-fashion-clip", # model name
    "modelProperties": {
        "name": "ViT-B-16", # model architecture
        "dimensions": 512, # embedding dimensions
        "url": "https://marqo-gcl-public.s3.us-west-2.amazonaws.com/marqo-fashionCLIP/marqo_fashionCLIP.pt", # model weights
        "type": "open_clip" # loading library
    },
}

api_key = "your_api_key"  # replace with your api key (https://www.marqo.ai/blog/finding-my-marqo-api-key)
mq = marqo.Client("https://api.marqo.ai", api_key=api_key)

mq.create_index("fashion-index", settings_dict=settings)

# triggers model download
mq.index("fashion-index").search("black dress")

```

See the [full documentation](https://docs.marqo.ai/2.11/#multi-modal-and-cross-modal-search) for more details on adding documents and searching.

## Quick Start
Install PyTorch first and run 
```bash
pip install -r requirements.txt
```

To evaluate Marqo-FashionCLIP, run this command
```bash
python eval.py \
        --dataset-config ./configs/${DATASET}.json \
        --model-name Marqo/marqo-fashionCLIP \
        --run-name Marqo-FashionCLIP
```
- `DATASET` can be one of ['deepfashion_inshop', 'deepfashion_multimodal', 'fashion200k', 'KAGL', 'atlas', 'polyvore' 'iMaterialist']

To evaluate Marqo-FashionSigLIP, run this command
```bash
python eval.py \
        --dataset-config ./configs/${DATASET}.json \
        --model-name Marqo/marqo-fashionSigLIP \
        --run-name Marqo-FashionSigLIP
```
- `DATASET` can be one of ['deepfashion_inshop', 'deepfashion_multimodal', 'fashion200k', 'KAGL', 'atlas', 'polyvore' 'iMaterialist']

Scripts to evaluate other models including [FashionCLIP 2.0](https://github.com/patrickjohncyh/fashion-clip) and [OpenFashionCLIP](https://github.com/aimagelab/open-fashion-clip) can be found in [scripts](scripts) directory.

## Datasets
We collected 7 public multimodal fashion datasets and uploaded to HuggingFace: [Atlas](https://huggingface.co/datasets/Marqo/atlas), [DeepFashion (In-shop)](https://huggingface.co/datasets/Marqo/deepfashion-inshop), [DeepFashion (Multimodal)](https://huggingface.co/datasets/Marqo/deepfashion-multimodal), [Fashion200k](https://huggingface.co/datasets/Marqo/fashion200k), [iMaterialist](https://huggingface.co/datasets/Marqo/iMaterialist), [KAGL](https://huggingface.co/datasets/Marqo/KAGL), and [Polyvore](https://huggingface.co/datasets/Marqo/polyvore). Each dataset has different metadata available. Thus, tasks for each dataset are stored as json files in [scripts](scripts) directory. Refer to our [blog](https://www.marqo.ai/blog/search-model-for-fashion) for more information about each dataset.

## Summarizing Results
To renew [LEADERBOARD.md](LEADERBOARD.md) and summarize results of different models locally, run this command
```bash
python summarize_results.py
```

## Citation
```
@software{Jung_Marqo-FashionCLIP_and_Marqo-FashionSigLIP_2024,
author = {Jung, Myong Chol and Clark, Jesse},
month = aug,
title = {{Marqo-FashionCLIP and Marqo-FashionSigLIP}},
url = {https://github.com/marqo-ai/marqo-FashionCLIP},
version = {1.0.0},
year = {2024}
}
```
