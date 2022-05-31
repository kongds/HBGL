#!/usr/bin/env bash
seed=42
TRAIN_FILE=./data/rcv1/rcv1_train_all_generated.json
OUTPUT_DIR=models/rcv1
CACHE_DIR=.cache
CUDA_VISIBLE_DEVICES=0 python run.py \
  --train_file ${TRAIN_FILE} --output_dir ${OUTPUT_DIR} \
  --model_type bert --model_name_or_path bert-base-uncased \
  --do_lower_case --max_source_seq_length 492 --max_target_seq_length 5 \
  --per_gpu_train_batch_size 12 --gradient_accumulation_steps 1 \
  --valid_file ./data/rcv1/rcv1_val_all_generated.json \
  --add_vocab_file ./data/rcv1/label_map.pkl \
  --label_smoothing 0 \
  --learning_rate 3e-5 --num_warmup_steps 500 --num_training_steps 96000 --cache_dir ${CACHE_DIR} \
  --test_file ./data/rcv1/rcv1_test_all_generated.json \
  --save_steps 3000 \
  --random_prob 0 --keep_prob 0 --soft_label --seed $seed --random_label_init \
 --label_cpt ./data/rcv1/rcv1.taxonomy   --label_cpt_steps 100 --rcv1_expand  ./data/rcv1/rcv1.topics.hier.expanded --label_cpt_use_bce
