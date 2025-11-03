import requests
from typing import List
import os
import json
import dotenv
dotenv.load_dotenv()

class Hyperbolic:
    def __init__(self, url, model):
        self.url = url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('HYPERBOLIC_API_KEY')}"
        }
        self.model = model
    def return_last_prompt_token_logprobs(self, prompts) -> List[float] | None:
        try:
            data = {
                "prompt": prompts,
                "model": self.model,
                "max_tokens": 0,
                "temperature": 0.7,
                "top_p": 0.9,
                # OpenAI-style APIs expect an integer here, not boolean
                "logprobs": 1,
                "echo": True,
            }
            response = requests.post(self.url, headers=self.headers, json=data)
            logprobs = []
            for choice in response.json()['choices']:
                logprobs.append(choice['logprobs']['token_logprobs'][-1])
            return logprobs
        except Exception as e:
            print(e)
            return self.return_last_prompt_token_logprobs(prompts)
    def return_first_generated_token_logprobs(self, prompts) -> List[float] | None:
        try:
            data = {
                "prompt": prompts,
                "model": self.model,
                "max_tokens": 1,
                "temperature": 0.7,
                "top_p": 0.9,
                "logprobs": 20
            }
            response = requests.post(self.url, headers=self.headers, json=data)
            logprobs = []
            for choice in response.json()['choices']:
                logprobs.append(choice['logprobs']['top_logprobs'][0])
            return logprobs
        except Exception as e:
            print(e)
            return None
    def return_chat_completion(self, prompts):
        chat_completions = []
        for prompt in prompts:
            message = {
                "role": "user",
                "content": f"{prompt} [True/False]. Return either True or False."
            }
            data = {
                "messages": [message],
                "model": self.model,
                "max_tokens": 1,
                "temperature": 0.7,
                "top_p": 0.9,
            }
            response = requests.post(self.url, headers=self.headers, json=data)
            chat_completions.append(response.json()['choices'][0]['message']['content'])
        return chat_completions