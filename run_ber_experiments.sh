#!/bin/bash
MODEL="meta-llama/Llama-3.2-1B"
DATASET="mnli"

LOSS_RATES=("0.0" "0.001" "0.005" "0.01")
NUM_NODES=("2" "10")
SEEDS=("10" "20" "30" "40" "50")
# GPU settings
export CUDA_VISIBLE_DEVICES=1
export WANDB_PROJECT="lossy_network"

# Create output directory if it doesn't exist
mkdir -p output

for loss_rate in "${LOSS_RATES[@]}"; do
  for nodes in "${NUM_NODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_id="${nodes}nodes_${DATASET}_lr${loss_rate}_seed${seed}"
      output_dir="output/${DATASET}"
      echo "Starting experiment: $run_id"
      # Run the experiment
      python src/main.py \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --loss_rate "$loss_rate" \
        --num_nodes "$nodes" \
        --batch_size $((16 * ${nodes})) \
        --learning_rate 2e-5 \
        -nunf 3 \
        --run_id "$run_id" \
        --epochs 7 \
        --seed "$seed" \
        --output_dir "$output_dir"
      
      echo "Completed experiment: $run_id"
      echo "--------------------------------"
    done
  done
done

echo "All experiments completed!" 
