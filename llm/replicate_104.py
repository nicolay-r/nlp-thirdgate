import logging

from bulk_chain.core.llm_base import BaseLM
from replicate import Client


class Replicate(BaseLM):

    LLaMA3_instruct_prompt_template = (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                                       "{template}<|eot_id|><|start_header_id|>user<|end_header_id|>"
                                       "\n\n{{prompt}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")

    @staticmethod
    def get_template(max_tokens, temp, top_k=50, template=""):
        return {
            "meta/llama-4-maverick-instruct": {
                "top_k": top_k,
                "top_p": 1.0,
                "length_penalty": 1,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "temperature": 0.1 if temp is None else temp,
                "max_tokens": min(max_tokens, 128000) if max_tokens is not None else 128000,
                "prompt_template": template,
            },
            "deepseek-ai/deepseek-r1": {
                "top_p": 1.0,                                        # Default parameter at replicate.com demo
                "frequency_penalty": 0,                              # Default parameter at replicate.com demo
                "presence_penalty": 0,                               # Default parameter at replicate.com demo
                "temperature": 0.6 if temp is None else temp,        # According to the DeepSeek documentation.
                "max_tokens": min(max_tokens, 20480) if max_tokens is not None else 20480,
                "prompt_template": "",                               # According to the DeepSeek documentation.
            },
            "meta/meta-llama-3-70b-instruct": {
                "top_k": top_k,
                "min_tokens": 0,
                "presence_penalty": 1.15,
                "frequency_penalty": 0.2,
                "temperature": 0.1 if temp is None else temp,
                "max_tokens": min(max_tokens, 4096) if max_tokens is not None else 4096,
                "prompt_template": Replicate.LLaMA3_instruct_prompt_template.format(template=template)
            },
            "meta/meta-llama-3-8b-instruct": {
                "top_k": top_k,
                "top_p": 0.9,
                "length_penalty": 1,
                "presence_penalty": 1.15,
                "temperature": 0.1 if temp is None else temp,
                "max_tokens": min(max_tokens, 4096) if max_tokens is not None else 4096,
                "prompt_template": Replicate.LLaMA3_instruct_prompt_template.format(template=template),
            }, 
        }

    def __init__(self, model_name, temp=None, max_tokens=None, api_token=None,
                 suppress_httpx_log=True, assistant="You are a helpful assistant", **kwargs):
        super(Replicate, self).__init__(name=model_name, **kwargs)
        self.r_model_name = model_name

        all_settings = self.get_template(max_tokens=max_tokens, temp=temp, template=assistant)

        if model_name not in all_settings:
            raise Exception(f"There is no predefined settings for `{model_name}`. Please, Tweak the model first!")

        self.settings = all_settings[model_name]
        self.client = Client(api_token=api_token)

        if suppress_httpx_log:
            httpx_logger = logging.getLogger("httpx")
            httpx_logger.setLevel(logging.WARNING)

    def ask(self, prompt):
        chunks = self.client.run(self.r_model_name, input=self.settings | {"prompt": prompt})
        return "".join(chunks)

    def ask_stream(self, prompt):
        chunks_it = self.client.stream(self.r_model_name, input=self.settings | {"prompt": prompt})
        return chunks_it

    async def ask_async(self, prompt):
        chunks = self.client.async_run(self.r_model_name, input=self.settings | {"prompt": prompt})
        return ''.join([str(chunk) for chunk in await chunks])

    async def ask_stream_async(self, prompt):
        chunks_it = self.client.async_stream(self.r_model_name, input=self.settings | {"prompt": prompt})
        return await chunks_it
