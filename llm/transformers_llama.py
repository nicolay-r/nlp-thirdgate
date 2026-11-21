# This provider is for bulk-chain 1.3.1
import torch
from transformers import pipeline
from bulk_chain.core.llm_base import BaseLM


class Llama32(BaseLM):
    """
        This code has been tested under transformers==4.47.0

        This is an experimental version of the LLaMA-3.2
        that has support of the batching mode.

        Tested under:
            transformers==4.44.2
            bulk-chain==1.3.1
    """

    def __init__(self, model_name, api_token=None, temp=0.1, device='cpu',
                 max_new_tokens=32768, use_bf16=False, **kwargs):
        super(Llama32, self).__init__(**kwargs)

        self.__max_new_tokens = max_new_tokens
        self.__pipe = pipeline("text-generation",
                               model=model_name,
                               torch_dtype=torch.bfloat16 if use_bf16 else "auto",
                               device_map=device,
                               temperature=temp,
                               pad_token_id=128001,
                               token=api_token)

    def ask(self, prompt, **kwargs):
        messages = [
            {"role": "user", "content": prompt}
        ]
        outputs = self.__pipe(messages, max_new_tokens=self.__max_new_tokens)
        return outputs[0]["generated_text"][-1]["content"]

    def ask_batch(self, batch):
        messages = [
            {"role": "user", "content": prompt}
            for prompt in batch
        ]
        outputs = self.__pipe(messages, max_new_tokens=self.__max_new_tokens, batch_size=len(messages))
        return [out["generated_text"][-1]["content"] for out in outputs]

    def ask_stream(self, prompt, **kwargs):
        yield self.ask(prompt, **kwargs)

    async def ask_async(self, prompt, **kwargs):
        return self.ask(prompt, **kwargs)

    async def ask_stream_async(self, prompt, **kwargs):
        return self.ask(prompt, **kwargs)

    async def ask_batch_async(self, batch):
        return self.ask_batch(batch)
