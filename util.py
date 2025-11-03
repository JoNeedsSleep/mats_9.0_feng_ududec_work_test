from typing import Any
import json
import pprint
import random
from typing import List
import matplotlib.pyplot as plt
import copy
import math
from statistics import mean, stdev
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def initialize_D(train_data, N=8):
    # list of dictionaries with keys 'question', 'choice', 'label', 'consistency_id', and a random choice 'y'
    random_samples = random.sample(list[Any](train_data), N)
    samples_with_labels = []
    for sample in random_samples:
        sample['y'] = random.choice(["True", "False"])
        samples_with_labels.append(sample)
    return samples_with_labels

def D_to_D_golden(D, train_data):
    D_golden = copy.deepcopy(D)
    # find the sample in train_data with the same choice and then take the label of that sample and convert it to a string "True" or "False"
    for sample in D_golden:
        for sample_train in train_data:
            if sample_train['choice'] == sample['choice']:
                sample['y'] = "True" if sample_train['label'] == 1 else "False"
    return D_golden

def retrieve_question_choices_labels(D):
    return [(sample['question'], sample['choice'], sample['y']) for sample in D]

def prep_prompt(D, question_to_evaluate, choice_to_evaluate, label_to_evaluate, fixed_context=None):
    prompt = ""
    if fixed_context is not None:
        return fixed_context + f"Question: {question_to_evaluate}\nClaim: {choice_to_evaluate}\nI think the claim is"
    for question, choice, y in retrieve_question_choices_labels(D):
        prompt += f"Question: {question}\nClaim: {choice}\nI think the claim is {y}\n"
    prompt += f"Question: {question_to_evaluate}\nClaim: {choice_to_evaluate}\nI think the claim is {label_to_evaluate}"
    #print(prompt)
    return prompt

def get_datastructure_for_comparison(D, train_data, chat_template):
    with open('result/chat_accuracy.json', 'r') as f:
        chat_accuracy = json.load(f)
    data = {
        'ICM': {
            'Context': D,
            'accuracy': 0,
            'prompts': []
            },
        'golden': {
            'Context': D_to_D_golden(D, train_data),
            'accuracy': 0,
            'prompts': []
        },
        'zero_shot': {
            'Context': chat_template,
            'accuracy': 0,
            'prompts': []
        },
        'zero_shot_chat': {
            'Context': None,
            'accuracy': chat_accuracy,
            'prompts': []
        }
    }
    return data

def calculate_true_logprob_tot(logprobs: List[dict]) -> List[float]:
    # loop thru & convert all tokens to lowercase and check if it contains false
    logprob_tot_values = []
    for logprob in logprobs:
        logprob_tot_value = 0
        for token, value in logprob.items():
            if 'true' in token.lower() or 'correct' in token.lower() or 'right' in token.lower():
                logprob_tot_value += value
        logprob_tot_values.append(-100 if logprob_tot_value == 0 else logprob_tot_value)
    return logprob_tot_values

def calculate_false_logprob_tot(logprobs: List[dict]) -> List[float]:
    # loop thru & convert all tokens to lowercase and check if it contains false
    logprob_tot_values = []
    for logprob in logprobs:
        logprob_tot_value = 0
        for token, value in logprob.items():
            if 'false' in token.lower() or 'incorrect' in token.lower() or 'wrong' in token.lower():
                logprob_tot_value += value
        logprob_tot_values.append(-100 if logprob_tot_value == 0 else logprob_tot_value)
    return logprob_tot_values

def get_test_score(test_data,list_of_true_probs,list_of_false_probs):
    right_ans = 0
    for i in range(len(test_data)):
        label=0
        if list_of_true_probs[i] > list_of_false_probs[i]:
            label=1
        if label==test_data[i]['label']:
            right_ans += 1
    return (right_ans / len(test_data))

def get_test_score_chat(chat_completions, test_data):
    right_ans = 0
    for i in range(len(test_data)):
        if 'true' in chat_completions[i].lower():
            label=1
        elif 'false' in chat_completions[i].lower():
            label=0 
        if label==test_data[i]['label']:
            right_ans += 1
    return (right_ans / len(test_data))

def plot_truthfulqa_from_jsonl(jsonl_path: str = 'result/comparison_data.jsonl', output_path: str | None = 'result/truthfulqa_plot.png', title: str = 'TruthfulQA', confidence: float = 0.95) -> None:
    """
    Load accuracies from a JSONL file where each line is a JSON object with
    keys ['zero_shot', 'zero_shot_chat', 'ICM', 'golden'] and values in [0,1].
    Compute mean accuracies and confidence intervals across runs, and plot.
    """
    order = ['zero_shot', 'zero_shot_chat', 'ICM', 'golden']
    labels = ['Zero-shot', 'Zero-shot (chat)', 'ICM', 'Golden']

    per_method_values = {k: [] for k in order}
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for k in order:
                if k in obj and obj[k] is not None:
                    try:
                        per_method_values[k].append(float(obj[k]))
                    except (TypeError, ValueError):
                        pass

    # Compute mean and symmetric CI using normal approximation on runs
    def z_for_conf(conf: float) -> float:
        # Two-sided normal quantile approximation
        if conf >= 0.999:
            return 3.29
        if conf >= 0.995:
            return 2.81
        if conf >= 0.99:
            return 2.58
        if conf >= 0.975:
            return 1.96
        if conf >= 0.95:
            return 1.96
        if conf >= 0.90:
            return 1.64
        return 1.96

    z = z_for_conf(confidence)
    means = []
    ci_margins = []
    counts = []
    for k in order:
        vals = per_method_values[k]
        n = len(vals)
        counts.append(n)
        if n == 0:
            means.append(0.0)
            ci_margins.append(0.0)
            continue
        m = mean(vals)
        if n > 1:
            s = stdev(vals)
            margin = z * (s / math.sqrt(n))
        else:
            margin = 0.0
        # Clip to [0,1] bounds in yerr later via min/max. Keep margin non-negative.
        means.append(m)
        ci_margins.append(max(0.0, margin))

    # Convert to percentages
    values_pct = [round(m * 100, 2) for m in means]
    yerr_pct = [round(margin * 100, 2) for margin in ci_margins]

    colors = ['#B57BA6', '#B57BA6', '#79BDCB', '#E7B35B']

    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=150)
    bars = ax.bar(range(len(values_pct)), values_pct, color=colors, edgecolor='black', linewidth=1, yerr=yerr_pct, capsize=3)
    bars[1].set_hatch('..')

    ax.set_title(title, fontsize=14, pad=6)
    ax.set_xticks(range(len(values_pct)))
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy (%)', fontsize=10)
    ax.set_ylim(0, 100)

    # Thin axes spines for a clean look
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    # Annotate mean and N on top of bars
    for bar, val, n in zip(bars, values_pct, counts):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1,
                "", ha='center', va='bottom', fontsize=8)

    fig.subplots_adjust(bottom=0.22)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()