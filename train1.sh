#!/bin/bash
set -x

export MODEL_TO_FINETUNE="meta-llama/Llama-3.2-1B"
export DATASET_PATH="./binary_data.jsonl"
export OUTPUT_SAVE_PATH="./checkpoint/binary_head_adapter"

# instruction prompt
export INSTRUCTION_TEMPLATE="You are a multilingual text moderation model.
Classify each input as either HARMFUL or SAFE.

HARMFUL means the text contains or promotes any form of violence, crime, sexual or child exploitation, hate, defamation, self-harm, harassment, privacy violation, or other abusive or unsafe behavior.
All other text is SAFE.

Always output only one word:
HARMFUL
or
SAFE

Text: {}
Label: "

# training command
training_commands="deepspeed --module openrlhf.cli.train_sft \
  --pretrain \"$MODEL_TO_FINETUNE\" \
  --dataset \"$DATASET_PATH\" \
  --input_key \"text\" \
  --output_key \"label\" \
  --input_template \"$INSTRUCTION_TEMPLATE\" \
  --save_path \"$OUTPUT_SAVE_PATH\" \
  --max_len 2048 \
  --max_epochs 1 \
  --zero_stage 3 \
  --bf16 \
  --learning_rate 5e-6 \
  --max_samples 1000000 \
  --train_batch_size 128 \
  --micro_train_batch_size 1 \
  --logging_steps 1 \
  --save_steps -1 \
  --eval_steps -1 \
  --attn_implementation \"flash_attention_2\" \
  --adam_offload \
  --packing_samples \
  --load_in_4bit \
  --lora_rank 64 \
  --lora_alpha 64 \
  --target_modules \""q_proj"\" \""k_proj"\" \""v_proj"\" \""o_proj"\" \""gate_proj"\" \""up_proj"\" \""down_proj"\""

# execute training
eval "$training_commands"
