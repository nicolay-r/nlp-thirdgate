import json

import aiohttp
import requests
from bulk_chain.core.llm_base import BaseLM


class OpenRouter(BaseLM):

    def __init__(self, model_name, api_token=None, **kwargs):
        super(OpenRouter, self).__init__(name=model_name, **kwargs)
        self.model = model_name
        self.token = api_token

    def _get_post_content(self, prompt):
        return {
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "headers": {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            },
            "data": json.dumps({
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
        }

    def ask(self, prompt):
        response = requests.post(**self._get_post_content(prompt))
        content_dict = json.loads(response.content)
        return content_dict['choices'][0]['message']['content'].strip()

    async def ask_async(self, prompt):
        async with (aiohttp.ClientSession() as session):
            async with session.post(**self._get_post_content(prompt)) as response:
                response.raise_for_status()
                content_dict = await response.json()
                return content_dict['choices'][0]['message']['content'].strip()
