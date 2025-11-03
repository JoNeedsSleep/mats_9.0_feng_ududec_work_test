#!/bin/bash
PYTHONPATH=$PYTHONPATH:$(pwd)

# Activate virtual environment
source icm/bin/activate

# so Hyperbolic becomes incredibly slow after about 80 iterations, but I noticed that we get pretty similar results on this task given a relatively small number of iterations, so I'm running the script 5 times, each time with 50 iterations (randomly sampled from the training set), and then plotting the results with a confidence interval.
for i in {1..5}; do
    python3 main.py \
        --iteration_num 50 \
        --url https://api.hyperbolic.xyz/v1/completions \
        --model meta-llama/Meta-Llama-3.1-405B
done

python3 plot_results.py \
    --jsonl_path result/comparison_data.jsonl \
    --output_path result/truthfulqa_plot.png \
    --title TruthfulQA