#!/bin/bash
PYTHONPATH=$PYTHONPATH:$(pwd)

# Activate virtual environment
source icm/bin/activate

python3 main.py \
    --iteration_num 250 \
    --url https://api.hyperbolic.xyz/v1/completions \
    --model meta-llama/Meta-Llama-3.1-405B

python3 plot_results.py --json_path result/comparison_data.json --output_path result/truthfulqa_plot.png --title TruthfulQA