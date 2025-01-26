import json

import requests
from bulk_chain.core.llm_base import BaseLM


class OpenRouter(BaseLM):

    def __init__(self, model_name, api_token=None, **kwargs):
        super(OpenRouter, self).__init__(name=model_name, **kwargs)
        self.model = model_name
        self.token = api_token

    def ask(self, prompt):
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.token}",
            },
            data=json.dumps({
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
        )

        content_dict = json.loads(response.content)
        return content_dict['choices'][0]['message']['content'].strip()
