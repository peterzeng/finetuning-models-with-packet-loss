#!/bin/bash
#MODEL="meta-llama/Llama-3.2-1B"
MODEL="openai-community/gpt2-large"
MODEL_ALIAS="gpt2-large"
DATASET="mnli"

LOSS_RATES=("0.0" "0.001" "0.005" "0.01")
# LOSS_RATES=("0.002")
# NUM_NODES=("2" "10")
NUM_NODES=("8")
SEEDS=("10" "20" "30" "40" "50")
#SEEDS=("10")
# GPU settings
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="new_lossy_network"

# Create output directory if it doesn't exist
mkdir -p output

for loss_rate in "${LOSS_RATES[@]}"; do
  for nodes in "${NUM_NODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_id="${MODEL_ALIAS}_${nodes}nodes_${DATASET}_lr${loss_rate}_seed${seed}"
      output_dir="${MODEL_ALIAS}_output/${DATASET}"
      echo "Starting experiment: $run_id"
      # Run the experiment
      python src/main.py \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --loss_rate "$loss_rate" \
        --num_nodes "$nodes" \
        --batch_size $((16 * ${nodes})) \
        --learning_rate 2e-5 \
        --run_id "$run_id" \
        --epochs 7 \
        --seed "$seed" \
        --output_dir "$output_dir" \
	--eval_steps 20 \
      
      echo "Completed experiment: $run_id"
      echo "--------------------------------"
    done
  done
done

echo "All experiments completed!" 
