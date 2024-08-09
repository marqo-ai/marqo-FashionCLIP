MODEL_NAME=OpenFashionCLIP
# Download this first by 'wget https://github.com/aimagelab/open-fashion-clip/releases/download/open-fashion-clip/finetuned_clip.pt'
PRETRAINED=./finetuned_clip.pt

for DATASET in 'deepfashion_inshop' 'deepfashion_multimodal' 'fashion200k' 'KAGL' 'atlas' 'polyvore' 'iMaterialist'
do
    python eval.py \
        --dataset-config ./configs/${DATASET}.json \
        --model-name ${MODEL_NAME} \
        --pretrained ${PRETRAINED} \
        --run-name OpenFashionCLIP \
        --query-prefix 'a photo of a '
done