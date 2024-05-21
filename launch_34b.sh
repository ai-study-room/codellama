!#/bin/bash

set -x

torchrun --nproc_per_node 4 example_instructions.py \
    --ckpt_dir CodeLlama-34b-Instruct/ \
    --tokenizer_path CodeLlama-34b-Instruct/tokenizer.model \
    --max_seq_len 512 --max_batch_size 4
