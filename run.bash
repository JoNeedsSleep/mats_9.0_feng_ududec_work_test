#!/bin/bash
PYTHONPATH=$PYTHONPATH:$(pwd)

# Activate virtual environment
source icm/bin/activate

for i in {1..5}; do
    python3 main.py \
        --iteration_num 100 \
        --url https://api.hyperbolic.xyz/v1/completions \
        --model meta-llama/Meta-Llama-3.1-405B
done

python3 plot_results.py \
    --jsonl_path result/comparison_data.jsonl \
    --output_path result/truthfulqa_plot.png \
    --title TruthfulQA