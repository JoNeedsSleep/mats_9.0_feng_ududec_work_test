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
from util import load_data, prep_prompt, retrieve_question_choices_labels, initialize_D
import random
import math
import argparse
from typing import Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device='cuda')
    return model, tokenizer

def get_log_prob(D, train_data, question_to_evaluate, choice_to_evaluate, label_to_evaluate, model, tokenizer):
    prompt = prep_prompt(D, train_data, question_to_evaluate, choice_to_evaluate, label_to_evaluate)
    text = prompt + choice_to_evaluate
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model.device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits 
        context_logits = logits[0, -2, :]
        last_token_id = input_ids[0, -1]
        log_probs = F.log_softmax(context_logits, dim=-1)
        last_log_prob = log_probs[last_token_id].item()
    return last_log_prob

def get_mutual_predictability_scores(D, train_data, model, tokenizer):
    sum_log_prob = 0
    for sample in D:
        D_setminus = D.copy()
        D_setminus.remove(sample)
        log_prob = get_log_prob(D_setminus, train_data, sample['question'], sample['choice'], sample['y'], model, tokenizer)
        sum_log_prob += log_prob
    return sum_log_prob

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial_temperature', type=float, default=10)
    parser.add_argument('--final_temperature', type=float, default=0.01)
    parser.add_argument('--iteration_num', type=int, default=300)
    parser.add_argument('--temperature_cooling_rate', type=float, default=0.99)
    parser.add_argument('--train_data', type=str, default='data/truthfulqa_train.json')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-405B') # this is the base model
    return parser.parse_args()

def main(args):
    train_data = load_data(args.train_data)
    initial_temperature = args.initial_temperature
    final_temperature = args.final_temperature
    iteration_num = args.iteration_num
    temperature_cooling_rate = args.temperature_cooling_rate
    temperature = initial_temperature
    print(len(train_data))
    model, tokenizer = load_model(args.model)
    # D is a dictionary of consistency_ids : choices_index pairs
    D = initialize_D(train_data)
    print(D)
    print(retrieve_question_choices_labels(D))

    for i in range(1, iteration_num + 1):
        temperature = max(initial_temperature / (1 + math.log(i) * temperature_cooling_rate), final_temperature)
        sample = random.choice(train_data)
        log_prob_of_True = get_log_prob(D, train_data, sample['question'], sample['choice'], "True", model, tokenizer)
        log_prob_of_False = get_log_prob(D, train_data, sample['question'], sample['choice'], "False", model, tokenizer)
        if log_prob_of_True > log_prob_of_False:
            label_to_evaluate = "True"
        else:
            label_to_evaluate = "False" 
        D_hat = D.copy()
        D_hat.append({'question': sample['question'], 'choice': sample['choice'], 'label': sample['label'], 'y': label_to_evaluate})
        P_D = get_mutual_predictability_scores(D, train_data, model, tokenizer)
        P_D_hat = get_mutual_predictability_scores(D_hat, train_data, model, tokenizer)
        # P_D_hat = 1
        if P_D_hat - P_D > 0 or (random.random() < math.exp((P_D_hat - P_D) / temperature)):
            print(f"D updated to D_hat: {D} to {D_hat}")
            D = D_hat
            

if __name__ == "__main__":
    args = arg_parse()
    main(args)