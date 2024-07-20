#!/bin/bash

SPLIT="test"
MODEL_TYPE= qwen1.5-1.8b
MODEL_BASE= /zhaobai46d/share/bunny/models/Qwen1.5-1.8B
TARGET_DIR= bunny-lora-qwen1.5-1.8b-chat

python -m bunny.eval.model_vqa_mmmu \
    --model-path ./checkpoints-$MODEL_TYPE/$TARGET_DIR \
    --model-base $MODEL_BASE \
    --model-type $MODEL_TYPE \
    --data-path ./eval/mmmu/MMMU \
    --config-path ./eval/mmmu/config.yaml \
    --output-path ./eval/mmmu/answers_upload/$SPLIT/$TARGET_DIR.json \
    --split $SPLIT \
    --conv-mode qwen2_finetune

python eval/mmmu/eval.py --output-path eval/mmmu/answers_upload/test/$TARGET_DIR.json