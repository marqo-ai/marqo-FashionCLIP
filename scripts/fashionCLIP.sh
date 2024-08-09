MODEL_NAME='patrickjohncyh/fashion-clip'

for DATASET in 'deepfashion_inshop' 'deepfashion_multimodal' 'fashion200k' 'KAGL' 'atlas' 'polyvore' 'iMaterialist'
do
    python eval.py \
        --dataset-config ./configs/${DATASET}.json \
        --model-name ${MODEL_NAME} \
        --run-name FashionCLIP
done