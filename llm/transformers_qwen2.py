# This provider is for bulk-chain 1.3.1
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from bulk_chain.core.llm_base import BaseLM


class Qwen2(BaseLM):
    """ Tested under:
            transformers==4.44.2
            bulk-chain==1.3.1
    """

    def __init__(self, model_name, temp=0.1, device='cpu',
                 max_new_tokens=None, token=None, use_bf16=False, **kwargs):
        super(Qwen2, self).__init__(**kwargs)

        self.__max_new_tokens = max_new_tokens
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16 if use_bf16 else "auto", token=token)
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, torch_dtype=torch.bfloat16 if use_bf16 else "auto", token=token, padding_side="left")

        self.__temp = temp

        self.__pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def ask(self, prompt, **kwargs):
        return self.ask_batch([prompt])[0]

    def ask_batch(self, batch):

        messages = [[{"role": "user", "content": prompt}] for prompt in batch]

        generation_args = {
            "max_new_tokens": self.__max_new_tokens,
            "return_full_text": False,
            "temperature": self.__temp,
            "do_sample": True,
        }

        output = self.__pipe(messages, **generation_args)
        return [response[0]["generated_text"] for response in output]

    def ask_stream(self, prompt, **kwargs):
        yield self.ask(prompt, **kwargs)

    async def ask_async(self, prompt, **kwargs):
        return self.ask(prompt, **kwargs)

    async def ask_stream_async(self, prompt, **kwargs):
        return self.ask(prompt, **kwargs)

    async def ask_batch_async(self, batch):
        return self.ask_batch(batch)
