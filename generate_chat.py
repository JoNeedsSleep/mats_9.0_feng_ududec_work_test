from call_hyperbolic import Hyperbolic
import json
from util import get_test_score_chat, prep_prompt

chat_llm = Hyperbolic(url="https://api.hyperbolic.xyz/v1/chat/completions", model="meta-llama/Meta-Llama-3.1-405B-Instruct")

with open('data/truthfulqa_test.json', 'r') as f:
    test_data = json.load(f)

prompts = [prep_prompt(None, sample['question'], sample['choice'], None, fixed_context="") for sample in test_data]
accuracy = get_test_score_chat(chat_llm.return_chat_completion(prompts), test_data)
print(accuracy)

with open('result/chat_accuracy.json', 'w') as f:
    json.dump(accuracy, f)