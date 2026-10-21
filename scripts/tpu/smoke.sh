#!/bin/bash
# NOTE: libtpu parses environment variables named like its flags (e.g. WRAP -> --wrap), so the wrapper path uses COFI_WRAPPER.
# Smoke test of the 3-stage CoFi MRPC pipeline with caps (identical on GPU and TPU).
# Usage: smoke.sh <code_dir> <out_root>   (python env must be active)
# Caps: stage A 200 steps, stage B 300 steps (pruning active from step 100), stage C 200 steps; logging every 10 steps,
# one eval (full 408-example dev set) at the last step; seed 57; all other hyperparameters as in SELECTION.md.
CODE=$1; OUT=$2; COFI_WRAPPER=${COFI_WRAPPER:-$(dirname $(readlink -f $0))/log_wrapper.py}
cd $CODE
mkdir -p $OUT
COMMON="--task_name MRPC --model_name_or_path bert-base-uncased --do_train --do_eval --max_seq_length 128 --per_device_eval_batch_size 32 --overwrite_output_dir --save_steps 0 --evaluation_strategy steps --seed 57 --logging_steps 10"
echo "STAGE_A_START $(date -u +%FT%TZ)"
python3 $COFI_WRAPPER --output_dir $OUT/teacher --ex_name MRPC_teacher $COMMON \
  --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 --eval_steps 200 --max_steps 200 > $OUT/stageA.log 2>&1
echo "STAGE_A_EXIT $? $(date -u +%FT%TZ)"
python3 $COFI_WRAPPER --output_dir $OUT/prune --ex_name MRPC_sparsity0.95 $COMMON \
  --per_device_train_batch_size 32 --learning_rate 2e-5 --reg_learning_rate 0.01 --num_train_epochs 100 --eval_steps 300 --max_steps 300 \
  --pruning_type structured_heads+structured_mlp+hidden+layer --pretrained_pruned_model None --target_sparsity 0.95 \
  --freeze_embeddings --do_distill --do_layer_distill --distillation_path $OUT/teacher/best \
  --distill_ce_loss_alpha 0.1 --distill_loss_alpha 0.9 --distill_temp 2 --scheduler_type linear \
  --layer_distill_version 4 --prepruning_finetune_epochs 4 --lagrangian_warmup_epochs 20 --sparsity_epsilon 0.01 > $OUT/stageB.log 2>&1
echo "STAGE_B_EXIT $? $(date -u +%FT%TZ)"
python3 $COFI_WRAPPER --output_dir $OUT/ft --ex_name MRPC_sparsity0.95 $COMMON \
  --per_device_train_batch_size 64 --learning_rate 3e-5 --reg_learning_rate 0.01 --num_train_epochs 20 --eval_steps 200 --max_steps 200 \
  --pruning_type None --pretrained_pruned_model $OUT/prune --target_sparsity 0.95 \
  --freeze_embeddings --do_distill --do_layer_distill --distillation_path $OUT/teacher/best \
  --distill_ce_loss_alpha 0.1 --distill_loss_alpha 0.9 --distill_temp 2 --scheduler_type none \
  --layer_distill_version 4 --prepruning_finetune_epochs 4 --lagrangian_warmup_epochs 20 > $OUT/stageC.log 2>&1
echo "STAGE_C_EXIT $? $(date -u +%FT%TZ)"
echo SMOKE_DONE
