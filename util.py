from typing import Any
import json
import pprint
import random
from typing import List
import matplotlib.pyplot as plt
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
    D_golden = D.copy()
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
        print(label, test_data[i]['label'])
        if label==test_data[i]['label']:
            right_ans += 1
            print("right")
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

def plot_truthfulqa_from_json(json_path: str = 'result/comparison_data.json', output_path: str | None = 'result/truthfulqa_plot.png', title: str = 'TruthfulQA') -> None:
    """
    Load accuracies from comparison_data.json and render a 4-bar chart matching the
    style of the example figure. Values in JSON are expected in [0,1] and are
    converted to percentages for display.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    order = ['zero_shot', 'zero_shot_chat', 'ICM', 'golden']
    labels = ['Zero-shot', 'Zero-shot (chat)', 'ICM', 'Golden']
    values = [round(float(data[k]) * 100, 2) for k in order]

    colors = ['#B57BA6', '#B57BA6', '#79BDCB', '#E7B35B']

    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=150)
    bars = ax.bar(range(len(values)), values, color=colors, edgecolor='black', linewidth=1)
    # Add dotted hatch to the second bar to echo the example style
    bars[1].set_hatch('..')

    ax.set_title(title, fontsize=14, pad=6)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy (%)', fontsize=10)
    ax.set_ylim(0, 100)

    # Thin axes spines for a clean look
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    # Annotate values on top of bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1,
                f"{val:.0f}", ha='center', va='bottom', fontsize=9)

    # Extra bottom margin so rotated labels don't clip/overlap
    fig.subplots_adjust(bottom=0.22)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()