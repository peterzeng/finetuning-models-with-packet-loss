#!/bin/bash
MODEL="meta-llama/Llama-3.2-1B"
# MODEL="openai-community/gpt2-large"
# MODEL_ALIAS="gpt2-large"
MODEL_ALIAS="llama-3.2-1b"
DATASET="mnli"

# LOSS_RATES=("0.0" "0.001" "0.005" "0.01")
NUM_NODES=("2" "4" "8" "10")
SEEDS=("10" "20" "30" "40" "50")
CONFIGS=("one_percent" "half_percent" "point2_percent" "long_point1_percent" "short_1percent" "short_half_percent" "short_point_2percent" "short_point1_percent")

export CUDA_VISIBLE_DEVICES=3
export WANDB_PROJECT="new_lossy_network"

for config in "${CONFIGS[@]}"; do
  for nodes in "${NUM_NODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_id="ge_${MODEL_ALIAS}_${nodes}nodes_${DATASET}_lr_${config}_seed${seed}"
      output_dir="ge_${MODEL_ALIAS}_output/${DATASET}"
      echo "Starting experiment: $run_id"
      # Run the experiment
      python src/main.py \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --num_nodes "$nodes" \
        --batch_size $((16 * ${nodes})) \
        --learning_rate 2e-5 \
        --run_id "$run_id" \
        --epochs 4 \
        --seed "$seed" \
        --output_dir "$output_dir" \
	      --eval_steps 20 \
        --loss_type "g-e" \
        --ge_config "$config" 
      echo "Completed experiment: $run_id"
      echo "--------------------------------"
    done
  done
done

echo "All experiments completed!" 
