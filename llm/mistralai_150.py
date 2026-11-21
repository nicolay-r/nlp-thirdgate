# This provider is for bulk-chain 1.3.1
import logging

from bulk_chain.core.llm_base import BaseLM

from mistralai import Mistral


class MistralAI(BaseLM):

    def __init__(self, model_name, api_token, temp=0.1, max_tokens=None, client_kwargs=None, suppress_httpx_log=True, **kwargs):
        super(MistralAI, self).__init__(**kwargs)
        self.__client = Mistral(api_key=api_token)
        self.__max_tokens = max_tokens
        self.__temperature = temp
        self.__model_name = model_name
        self.__mistral_client_kwargs = {} if client_kwargs is None else client_kwargs

        if suppress_httpx_log:
            httpx_logger = logging.getLogger("httpx")
            httpx_logger.setLevel(logging.WARNING)

    def ask(self, prompt, **kwargs):

        response = self.__client.chat.complete(
            max_tokens=self.__max_tokens,
            temperature=self.__temperature,
            model=self.__model_name,
            messages=[{"role": "user", "content": prompt}],
            **self.__mistral_client_kwargs
        )

        return response.choices[0].message.content

    def ask_batch(self, batch):
        return [self.ask(prompt) for prompt in batch]

    def ask_stream(self, prompt, **kwargs):
        yield self.ask(prompt, **kwargs)

    async def ask_async(self, prompt, **kwargs):
        return self.ask(prompt, **kwargs)

    async def ask_stream_async(self, prompt, **kwargs):
        return self.ask(prompt, **kwargs)

    async def ask_batch_async(self, batch):
        return self.ask_batch(batch)
