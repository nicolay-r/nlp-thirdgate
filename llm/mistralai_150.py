from bulk_chain.core.llm_base import BaseLM

from mistralai import Mistral


class MistralAI(BaseLM):

    def __init__(self, model_name, api_token, temp=0.1, max_tokens=None, client_kwargs=None, suppress_httpx_log=True, **kwargs):
        super(MistralAI, self).__init__(name=model_name, **kwargs)

        # dynamic import of the OpenAI library.

        self.__client = Mistral(api_key=api_token)
        self.__max_tokens = max_tokens
        self.__temperature = temp
        self.__model_name = model_name
        self.__mistral_client_kwargs = {} if client_kwargs is None else client_kwargs

        if suppress_httpx_log:
            httpx_logger = logging.getLogger("httpx")
            httpx_logger.setLevel(logging.WARNING)

    def ask(self, prompt):

        response = self.__client.chat.complete(
            max_tokens=self.__max_tokens,
            temperature=self.__temperature,
            model=self.__model_name,
            messages=[{"role": "user", "content": prompt}],
            **self.__mistral_client_kwargs
        )

        return response.choices[0].message.content
