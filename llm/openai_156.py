# This code has been tested under openai==1.56.0
import logging
from openai import OpenAI
from bulk_chain.core.llm_base import BaseLM


class OpenAIProvider(BaseLM):

    def __init__(self, api_token, model_name, base_url="https://api.openai.com/v1", 
                 temp=0.1, max_tokens=None, assistant_prompt=None,
                 freq_penalty=0.0, attempts=None, suppress_httpx_log=True, **kwargs):
        assert (isinstance(assistant_prompt, str) or assistant_prompt is None)
        super(OpenAIProvider, self).__init__(name=model_name, attempts=attempts, **kwargs)

        # dynamic import of the OpenAI library.
        self.__client = OpenAI(api_key=api_token, base_url=base_url)
        self.__model_name = model_name
        self.__assistant_prompt = assistant_prompt if assistant_prompt is not None else None
        self.__kwargs = {} if kwargs is None else kwargs
        self.__kwargs |= {
            "frequency_penalty": freq_penalty,
            "max_tokens": 256 if max_tokens is None else max_tokens,
            "temperature": temp
        }

        # Optionally disable console logging.
        if suppress_httpx_log:
            httpx_logger = logging.getLogger("httpx")
            httpx_logger.setLevel(logging.WARNING)
            
    def __msg_content(self, prompt):
        if self.__assistant_prompt is not None:
            yield {
                "role": "assistant", 
                "content": self.__assistant_prompt
            }

        yield {
            "role": "user", 
            "content": prompt
        }
    
    def ask(self, prompt):
        response = self.__client.chat.completions.create(
            model=self.__model_name,
            messages=list(self.__msg_content(prompt)),
            **self.__kwargs)
        return response.choices[0].message.content

    def ask_stream(self, prompt):
        stream = self.__client.responses.create(
            model=self.__model_name,
            messages=list(self.__msg_content(prompt)),
            stream=True,
            **self.__kwargs)
        for chunk in stream:
            if chunk.type == 'response.output_text.delta':
                yield chunk.delta

    async def ask_async(self, prompt):
        response = await self.__client.responses.create(
            model=self.__model_name,
            input=list(self.__msg_content(prompt)),
            **self.__kwargs)
        return response.output_text