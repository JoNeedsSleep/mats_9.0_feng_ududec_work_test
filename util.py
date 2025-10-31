from typing import Any
import json
import pprint
import random

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


def retrieve_question_choices_labels(D):
    return [(sample['question'], sample['choice'], sample['y']) for sample in D]

def prep_prompt(D, train_data, question_to_evaluate, choice_to_evaluate, label_to_evaluate):
    prompt = ""
    for question, choice, y in retrieve_question_choices_labels(D):
        prompt += f"Question: {question}\nAnswer: {choice}\nI think the answer is {y}."
    prompt += f"Question: {question_to_evaluate}\nAnswer: {choice_to_evaluate}\nI think the answer is {label_to_evaluate}."
    #print(prompt)
    return prompt

if __name__ == "__main__":
    data_dict = load_data('data/truthfulqa_train.json')
    print(len(data_dict))
    pprint.pprint(data_dict[648][0]['question'])