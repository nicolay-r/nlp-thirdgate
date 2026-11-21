# This provider is for bulk-chain 1.3.1
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from bulk_chain.core.llm_base import BaseLM


class FlanT5(BaseLM):
    """ Tested under:
        transformers==4.44.2
        bulk-chain==1.3.1

        This implementation exploits batching mode.
    """

    def __init__(self, model_name="google/flan-t5-base", temp=0.1, device='cpu',
                 max_new_tokens=None, use_bf16=False, **kwargs):
        super(FlanT5, self).__init__(**kwargs)
        self.__device = device
        self.__model = T5ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16 if use_bf16 else None)
        self.__model.to(device)
        self.__tokenizer = AutoTokenizer.from_pretrained(
            model_name, torch_dtype=torch.bfloat16 if use_bf16 else None)
        self.__temp = temp
        self.__max_new_tokens = max_new_tokens

    def ask(self, prompt, **kwargs):
        return self.ask_batch([prompt])[0]

    def ask_batch(self, batch):
        inputs = self.__tokenizer(list(batch), return_tensors="pt", padding=True, truncation=True)
        inputs.to(self.__device)
        outputs = self.__model.generate(**inputs, max_new_tokens=self.__max_new_tokens,
                                        temperature=self.__temp, do_sample=True)
        return self.__tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def ask_stream(self, prompt, **kwargs):
        yield self.ask(prompt, **kwargs)

    async def ask_async(self, prompt, **kwargs):
        return self.ask(prompt, **kwargs)

    async def ask_stream_async(self, prompt, **kwargs):
        return self.ask(prompt, **kwargs)

    async def ask_batch_async(self, batch):
        return self.ask_batch(batch)
