#!/bin/bash

MODEL_TYPE=phi-2
OUTPUT_DIR=bunny-lora-$MODEL_TYPE-$2-imagevideo

mkdir -p ./checkpoints-$MODEL_TYPE/$OUTPUT_DIR

deepspeed bunny/train/train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./script/deepspeed/zero3.json \
    --model_name_or_path /zhaobai46d/share/bunny/models/phi-2-msft \
    --model_type $MODEL_TYPE \
    --version bunny \
    --is_multi_image True \
    --data_path /zhaobai46d/share/bunny/data/finetune/bunny_695k.json /zhaobai46d/share/shuyan/all_video_train.json  \
    --image_folder /zhaobai46d/share/bunny/data/finetune/images /zhaobai46d/share/shuyan/video_data \
    --vision_tower /zhaobai46d/share/bunny/models/siglip-so400m-patch14-384 \
    --pretrain_mm_mlp_adapter  /zhaobai46d/share/yexin/Bunny/checkpoints-pretrain/bunny-phi-2-pretrain-sppv2/mm_projector.bin \
    --mm_projector_type spp-v2 \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints-$MODEL_TYPE/$OUTPUT_DIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
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
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none | tee 2>&1 ./checkpoints-$MODEL_TYPE/$OUTPUT_DIR/log.txt \