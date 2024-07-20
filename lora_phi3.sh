#!/bin/bash

MODEL_TYPE=phi-3
OUTPUT_DIR=bunny-lora-$MODEL_TYPE-iou

mkdir -p ./checkpoints-$MODEL_TYPE/$OUTPUT_DIR

deepspeed bunny/train/train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./script/deepspeed/zero2.json \
    --model_name_or_path /zhaobai46d/share/bunny/models/Phi-3-mini-128k-instruct \
    --model_type $MODEL_TYPE \
    --version phi3 \
    --is_multi_image True \
    --data_path /zhaobai46d/dataset/videodata/qvhighlights/test4.json  \
    --image_folder /zhaobai46d/dataset/videodata/qvhighlights/videos  \
    --vision_tower /zhaobai46d/share/bunny/models/siglip-so400m-patch14-384 \
    --pretrain_mm_mlp_adapter  /zhaobai46d/videobunny/checkpoints-pretrain/bunny-phi-3-pretrain-25token/mm_projector.bin \
    --mm_projector_type c-abs \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints-$MODEL_TYPE/$OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 3000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none | tee 2>&1 ./checkpoints-$MODEL_TYPE/$OUTPUT_DIR/log.txt \