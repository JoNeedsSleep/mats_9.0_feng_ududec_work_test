"""
The idea for this implementation is to start with N = 8 randomly chosen samples D
from the training set \{x_i, y_i\}, and then for i = 1,...,iteration_num, we pick the answer
y for x_i with the highest log prob conditional on D, and compare the following two 
coherence/mutual predictability scores:
(1) P(D):= \sum_j^{|D|} log p(y_j | D \ (x_j,y_j))
(2) P(D + (x_i,y)):= \sum_j^{|D|+1} log p(y_j|x_j, (D + (x_i,y) \ (x_j,y_j))
If (2) - (1) > 0 we adopt this new label, if not we accept this label 
with probability exp(((2) - (1)) / temperature)
"""
from util import load_data, prep_prompt, retrieve_question_choices_labels, initialize_D, D_to_D_golden, get_datastructure_for_comparison, calculate_true_logprob_tot, calculate_false_logprob_tot, get_test_score
import random
import math
import argparse
from typing import Any, List
import os
from dotenv import load_dotenv
import json
from tqdm import tqdm
import copy
# Load environment variables from .env file
load_dotenv()

from call_hyperbolic import Hyperbolic


def get_mutual_predictability_scores(D_list, llm):
    prompts = []
    for D in D_list:
        for sample in D:
            D_setminus = copy.deepcopy(D)
            D_setminus.remove(sample)
            prompts.append(prep_prompt(D_setminus, sample['question'], sample['choice'], sample['y']))
    log_probs = llm.return_last_prompt_token_logprobs(prompts)
    if log_probs is None:
        return (0, 0)
    return sum(log_probs[:len(prompts)//2]), sum(log_probs[len(prompts)//2:])

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial_temperature', type=float, default=10)
    parser.add_argument('--final_temperature', type=float, default=0.01)
    parser.add_argument('--iteration_num', type=int, default=300)
    parser.add_argument('--temperature_cooling_rate', type=float, default=0.99)
    parser.add_argument('--train_data', type=str, default='data/truthfulqa_train.json')
    parser.add_argument('--test_data', type=str, default='data/truthfulqa_test.json')
    parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3.1-405B')
    parser.add_argument('--url', type=str, default='https://api.hyperbolic.xyz/v1/completions')
    parser.add_argument('--output_file', type=str, default='finetuning/ft_data.json')
    parser.add_argument('--chat_url', type=str, default='https://api.hyperbolic.xyz/v1/chat/completions')
    parser.add_argument('--chat_model', type=str, default='meta-llama/Meta-Llama-3.1-405B-Instruct')
    parser.add_argument('--chat_template_path', type=str, default='data/chat_template.txt')
    return parser.parse_args()

def main(args):
    train_data = load_data(args.train_data)
    initial_temperature = args.initial_temperature
    final_temperature = args.final_temperature
    iteration_num = args.iteration_num
    temperature_cooling_rate = args.temperature_cooling_rate
    temperature = initial_temperature
    llm = Hyperbolic(args.url, args.model)
    chat_llm = Hyperbolic(args.chat_url, args.chat_model)
    # D is a dictionary of consistency_ids : choices_index pairs
    D = initialize_D(train_data)

    for i in tqdm(range(1, iteration_num + 1)):
        temperature = max(initial_temperature / (1 + math.log(i) * temperature_cooling_rate), final_temperature)
        sample = random.choice(train_data)
        
        true_prompt = prep_prompt(D, sample['question'], sample['choice'], "True")
        false_prompt = prep_prompt(D, sample['question'], sample['choice'], "False")
        log_probs_of_True_False = llm.return_last_prompt_token_logprobs([true_prompt, false_prompt])
        if log_probs_of_True_False is None:
            continue
        if log_probs_of_True_False[0] > log_probs_of_True_False[1]:
            label_to_evaluate = "True"
        else:
            label_to_evaluate = "False"
        D_hat = copy.deepcopy(D)
        D_hat.append({'question': sample['question'], 'choice': sample['choice'], 'label': sample['label'], 'y': label_to_evaluate})
        P = get_mutual_predictability_scores([D, D_hat], llm)
        if P[1]==0 and P[0]==0:
            continue
        if P[1] - P[0] > 0 or (random.random() < math.exp((P[1] - P[0]) / temperature)):
            D = D_hat
            
    with open(args.output_file, 'w') as f:
        json.dump(D, f, indent=4)
    print(f"Output saved to {args.output_file}")

    with open(args.test_data, 'r') as f:
        test_data = json.load(f)
    with open(args.chat_template_path, 'r') as f:
        chat_template = f.read()
    
    comparison_data = get_datastructure_for_comparison(D, train_data, chat_template)
    for sample in tqdm(test_data, desc="Processing test data", total=len(test_data)):
        question = sample['question']
        choice = sample['choice']
        prompt_ICM = prep_prompt(D, question, choice, "")
        prompt_golden = prep_prompt(comparison_data['golden']['Context'], question, choice, "")
        prompt_zero_shot = prep_prompt(D, question, choice, "", fixed_context=chat_template)
        prompt_zero_shot_chat = prep_prompt(D, question, choice, "", fixed_context="")
        comparison_data['ICM']['prompts'].append(prompt_ICM)
        comparison_data['golden']['prompts'].append(prompt_golden)
        comparison_data['zero_shot']['prompts'].append(prompt_zero_shot)
        comparison_data['zero_shot_chat']['prompts'].append(prompt_zero_shot_chat)
    
    comparison_data['ICM']['log_probs'] = llm.return_first_generated_token_logprobs(comparison_data['ICM']['prompts'])
    comparison_data['golden']['log_probs'] = llm.return_first_generated_token_logprobs(comparison_data['golden']['prompts'])
    comparison_data['zero_shot']['log_probs'] = llm.return_first_generated_token_logprobs(comparison_data['zero_shot']['prompts'])
    comparison_data['ICM']['accuracy'] = get_test_score(test_data, calculate_true_logprob_tot(comparison_data['ICM']['log_probs']), calculate_false_logprob_tot(comparison_data['ICM']['log_probs']))
    print(comparison_data['ICM']['log_probs'])
    comparison_data['golden']['accuracy'] = get_test_score(test_data, calculate_true_logprob_tot(comparison_data['golden']['log_probs']), calculate_false_logprob_tot(comparison_data['golden']['log_probs']))
    comparison_data['zero_shot']['accuracy'] = get_test_score(test_data, calculate_true_logprob_tot(comparison_data['zero_shot']['log_probs']), calculate_false_logprob_tot(comparison_data['zero_shot']['log_probs']))
    print({key: comparison_data[key]['accuracy'] for key in comparison_data})
    with open('result/comparison_data.jsonl', 'a') as f:
        f.write(json.dumps({key: comparison_data[key]['accuracy'] for key in comparison_data}) + "\n")
if __name__ == "__main__":
    args = arg_parse()
    main(args)