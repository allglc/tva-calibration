for MODEL in 'vit_base_patch16_224' 'beit_large_patch16_224' 'mixer_b16_224' 'resnet50'
do

for SEED in 2020 2021 2022 2023 2024
do

python ./ProximityBias-Calibration-main/compute_calibration_metrics.py  \
    --model $MODEL \
    --distance_measure L2 \
    --data_dir "ProximityBias-Calibration-main/intermediate_output/imagenet/" \
    --random_seed $SEED  \
    --num_neighbors 10  

done
done