#!/usr/bin/env bash
seed=42
OUTPUT_DIR=models/nyt
CACHE_DIR=.cache
TRAIN_FILE=./data/nyt/nyt_train_all_generated_tl.json
CUDA_VISIBLE_DEVICES=0 python run.py \
  --train_file ${TRAIN_FILE} --output_dir ${OUTPUT_DIR} \
  --model_type bert --model_name_or_path bert-base-uncased \
  --do_lower_case --max_source_seq_length 472 --max_target_seq_length 9 \
  --per_gpu_train_batch_size 12 --gradient_accumulation_steps 1 \
  --valid_file ./data/nyt/nyt_val_all_generated.json \
  --test_file ./data/nyt/nyt_test_all_generated.json \
  --add_vocab_file ./data/nyt/label_map.pkl \
  --label_smoothing 0 \
  --wandb \
  --learning_rate 3e-5 --num_warmup_steps 500 --num_training_steps 96000 --cache_dir ${CACHE_DIR} \
  --random_prob 0 --keep_prob 0 --soft_label --seed ${seed}  \
  --label_cpt ./data/nyt/nyt.taxonomy --label_cpt_use_bce \
  --nyt_only_last_label_init \
  --label_cpt_not_incr_mask_ratio --label_cpt_steps 1000 --label_cpt_lr 1e-4
