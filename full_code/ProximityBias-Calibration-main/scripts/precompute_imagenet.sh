for MODEL in 'vit_base_patch16_224' 'beit_large_patch16_224' 'mixer_b16_224' 'resnet50'
do
for SEED in 2020 2021 2022 2023 2024
do
python ./ProximityBias-Calibration-main/pytorch-image-models/precompute_intermediate_results.py \
    --data_dir "/gpfsdswork/dataset/imagenet/val" \
    --dataset "imagenet" \
    --model $MODEL \
    --output_dir "ProximityBias-Calibration-main/intermediate_output/imagenet" \
    --split 'val' \
    --num-gpu 1
done
done