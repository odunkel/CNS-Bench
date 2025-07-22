#!/bin/bash

DATA_DIR=$1
WORK_DIR=$2
TAG=$3

### Supervised
bash benchmarks/run_cns_bench.sh resnet18.a1_in1k ${DATA_DIR} ${WORK_DIR}/resnet18.a1_in1k/${TAG}
bash benchmarks/run_cns_bench.sh resnet34.a1_in1k ${DATA_DIR} ${WORK_DIR}/resnet34.a1_in1k/${TAG}
bash benchmarks/run_cns_bench.sh resnet50.a1_in1k ${DATA_DIR} ${WORK_DIR}/resnet50.a1_in1k/${TAG}
bash benchmarks/run_cns_bench.sh resnet101.a1_in1k ${DATA_DIR} ${WORK_DIR}/resnet101.a1_in1k/${TAG}
bash benchmarks/run_cns_bench.sh resnet152.a1_in1k ${DATA_DIR} ${WORK_DIR}/resnet152.a1_in1k/${TAG}
bash benchmarks/run_cns_bench.sh vit_base_patch16_224.augreg_in1k ${DATA_DIR} ${WORK_DIR}/vit_base_patch16_224.augreg_in1k/${TAG}
bash benchmarks/run_cns_bench.sh vit_base_patch16_clip_224.openai_ft_in1k ${DATA_DIR} ${WORK_DIR}/vit_base_patch16_clip_224.openai_ft_in1k/${TAG}
bash benchmarks/run_cns_bench.sh vit_base_patch16_224.augreg_in21k_ft_in1k ${DATA_DIR} ${WORK_DIR}/vit_base_patch16_224.augreg_in21k_ft_in1k/${TAG}
bash benchmarks/run_cns_bench.sh deit_base_patch16_224.fb_in1k ${DATA_DIR} ${WORK_DIR}/deit_base_patch16_224.fb_in1k/${TAG}
bash benchmarks/run_cns_bench.sh deit3_small_patch16_224.fb_in1k ${DATA_DIR} ${WORK_DIR}/deit3_small_patch16_224.fb_in1k/${TAG}
bash benchmarks/run_cns_bench.sh deit3_medium_patch16_224.fb_in1k ${DATA_DIR} ${WORK_DIR}/deit3_medium_patch16_224.fb_in1k/${TAG}
bash benchmarks/run_cns_bench.sh deit3_base_patch16_224.fb_in1k ${DATA_DIR} ${WORK_DIR}/deit3_base_patch16_224.fb_in1k/${TAG}
bash benchmarks/run_cns_bench.sh deit3_large_patch16_224.fb_in1k ${DATA_DIR} ${WORK_DIR}/deit3_large_patch16_224.fb_in1k/${TAG}
bash benchmarks/run_cns_bench.sh deit3_huge_patch14_224.fb_in1k ${DATA_DIR} ${WORK_DIR}/deit3_huge_patch14_224.fb_in1k/${TAG}
bash benchmarks/run_cns_bench.sh convnext_tiny.fb_in1k ${DATA_DIR} ${WORK_DIR}/convnext_tiny.fb_in1k/${TAG}
bash benchmarks/run_cns_bench.sh convnext_small.fb_in1k ${DATA_DIR} ${WORK_DIR}/convnext_small.fb_in1k/${TAG}
bash benchmarks/run_cns_bench.sh convnext_base.fb_in1k ${DATA_DIR} ${WORK_DIR}/convnext_base.fb_in1k/${TAG}
bash benchmarks/run_cns_bench.sh convnext_large.fb_in1k ${DATA_DIR} ${WORK_DIR}/convnext_large.fb_in1k/${TAG}

### Self-Supervised
## MAE
bash benchmarks/run_cns_bench.sh vit_base_patch16 ${DATA_DIR} ${WORK_DIR}/mae_vit_base_patch16/${TAG} --global_pool --ckpt_path models/mae_finetuned_vit_base.pth  # available from the official mae github repo
bash benchmarks/run_cns_bench.sh vit_large_patch16 ${DATA_DIR} ${WORK_DIR}/mae_vit_large_patch16/${TAG} --global_pool --ckpt_path models/mae_finetuned_vit_large.pth  # available from the official mae github repo
bash benchmarks/run_cns_bench.sh vit_huge_patch14 ${DATA_DIR} ${WORK_DIR}/mae_vit_huge_patch14/${TAG} --global_pool --ckpt_path models/mae_finetuned_vit_huge.pth  # available from the official mae github repo
bash benchmarks/run_cns_bench.sh convnextv2_base.fcmae_ft_in1k ${DATA_DIR} ${WORK_DIR}/convnextv2_base.fcmae_ft_in1k/${TAG}
bash benchmarks/run_cns_bench.sh convnextv2_large.fcmae_ft_in1k ${DATA_DIR} ${WORK_DIR}/convnextv2_large.fcmae_ft_in1k/${TAG}
bash benchmarks/run_cns_bench.sh convnextv2_huge.fcmae_ft_in1k ${DATA_DIR} ${WORK_DIR}/convnextv2_huge.fcmae_ft_in1k/${TAG}
## MoCo v3
bash benchmarks/run_cns_bench.sh vit_base_patch16 ${DATA_DIR} ${WORK_DIR}/mocov3_vit_base_patch16/${TAG} --ckpt_path models/mocov3_finetuned_vit_base.pth  # available from the official mocov3 github repo
## DINO
bash benchmarks/run_cns_bench.sh dino_vit_base_patch16 ${DATA_DIR} ${WORK_DIR}/dino_vit_base_patch16/${TAG}
## DINO v2
bash benchmarks/run_cns_bench.sh dinov2_vit_small_patch14 ${DATA_DIR} ${WORK_DIR}/dinov2_vit_small_patch14/${TAG}
bash benchmarks/run_cns_bench.sh dinov2_vit_small_patch14_reg ${DATA_DIR} ${WORK_DIR}/dinov2_vit_small_patch14_reg/${TAG}
bash benchmarks/run_cns_bench.sh dinov2_vit_base_patch14 ${DATA_DIR} ${WORK_DIR}/dinov2_vit_base_patch14/${TAG}
bash benchmarks/run_cns_bench.sh dinov2_vit_base_patch14_reg ${DATA_DIR} ${WORK_DIR}/dinov2_vit_base_patch14_reg/${TAG}
bash benchmarks/run_cns_bench.sh dinov2_vit_large_patch14 ${DATA_DIR} ${WORK_DIR}/dinov2_vit_large_patch14/${TAG}
bash benchmarks/run_cns_bench.sh dinov2_vit_large_patch14_reg ${DATA_DIR} ${WORK_DIR}/dinov2_vit_large_patch14_reg/${TAG}
bash benchmarks/run_cns_bench.sh dinov2_vit_giant_patch14 ${DATA_DIR} ${WORK_DIR}/dinov2_vit_giant_patch14/${TAG}
bash benchmarks/run_cns_bench.sh dinov2_vit_giant_patch14_reg ${DATA_DIR} ${WORK_DIR}/dinov2_vit_giant_patch14_reg/${TAG}
### VLM
## CLIP
bash benchmarks/run_cns_bench.sh clip_resnet50 ${DATA_DIR} ${WORK_DIR}/clip_resnet50/${TAG}
bash benchmarks/run_cns_bench.sh clip_resnet101 ${DATA_DIR} ${WORK_DIR}/clip_resnet101/${TAG}
bash benchmarks/run_cns_bench.sh clip_vit_base_patch32_224 ${DATA_DIR} ${WORK_DIR}/clip_vit_base_patch32_224/${TAG}
bash benchmarks/run_cns_bench.sh clip_vit_base_patch16_224 ${DATA_DIR} ${WORK_DIR}/clip_vit_base_patch16_224/${TAG}
bash benchmarks/run_cns_bench.sh clip_vit_large_patch14_224 ${DATA_DIR} ${WORK_DIR}/clip_vit_large_patch14_224/${TAG}
bash benchmarks/run_cns_bench.sh clip_vit_large_patch14_336 ${DATA_DIR} ${WORK_DIR}/clip_vit_large_patch14_336/${TAG} --input-size 336 --crop-pct 1.0
