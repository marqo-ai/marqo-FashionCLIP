MODEL_NAME=ViT-B-16
PRETRAINED=laion2b_s34b_b88k

for DATASET in 'deepfashion_inshop' 'deepfashion_multimodal' 'fashion200k' 'KAGL' 'atlas' 'polyvore' 'iMaterialist'
do
    python eval.py \
        --dataset-config ./configs/${DATASET}.json \
        --model-name ${MODEL_NAME} \
        --pretrained ${PRETRAINED} \
        --run-name ViT-B-16-laion2b_s34b_b88k 
done