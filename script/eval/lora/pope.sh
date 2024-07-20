#!/bin/bash

MODEL_TYPE=qwen1.5-1.8b
MODEL_BASE=/zhaobai46d/share/bunny/models/Qwen1.5-1.8B
TARGET_DIR=bunny-lora-$MODEL_TYPE-bunny

python -m bunny.eval.model_vqa_loader \
    --model-path ./checkpoints-$MODEL_TYPE/$TARGET_DIR \
    --model-base $MODEL_BASE \
    --model-type $MODEL_TYPE \
    --question-file ./eval/pope/bunny_pope_test.jsonl \
    --image-folder ./eval/pope/val2014 \
    --answers-file ./eval/pope/answers/$TARGET_DIR.jsonl \
    --temperature 0 \
    --conv-mode bunny

python eval/pope/eval_pope.py \
    --annotation-dir ./eval/pope/coco \
    --question-file ./eval/pope/bunny_pope_test.jsonl \
    --result-file ./eval/pope/answers/$TARGET_DIR.jsonl
