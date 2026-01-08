#!/bin/bash
set -e




# export WANDB_API_KEY=your_wandb_api_key_here
TASK="vae_train"

# Unset distributed/SLURM variables to prevent implicit distributed setup
unset WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT
unset SLURM_PROCID SLURM_NTASKS SLURM_NPROCS SLURM_LOCALID

python vae/train_vae.py \
    --run_name vae_train \
    --model_name_or_path "meta-llama/Llama-3.2-3B" \
    --lora_r 512 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --output_dir "checkpoints" \
    --input_type "full_format" \
    --test_size 10 \
    --max_steps 30000 \
    --num_train_epochs 10 \
    --learning_rate 2e-5 \
    --lr_scheduler_type "cosine" \
    --lr_scheduler_kwargs '{"num_cycles": 1}' \
    --warmup_steps 1000 \
    --optim "adamw_torch" \
    --weight_decay 0.03 \
    --eval_strategy "no" \
    --eval_interval 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --notes "VAE training experiment" \
    # --ddp_backend "nccl"
    # --fsdp "hybrid_shard auto_wrap" \
    # --fsdp_config '{"backward_prefetch": "backward_pre", "forward_prefetch": true, "cpu_ram_efficient_loading": true, "sync_module_states": true, "transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"], "use_orig_params": true, "activation_checkpointing": false}' \
