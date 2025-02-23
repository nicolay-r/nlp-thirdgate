# This code has been tested under transformers==4.47.0

import torch
from transformers import pipeline
from bulk_chain.core.llm_base import BaseLM


class Llama32(BaseLM):

    def __init__(self, model_name, api_token=None, temp=0.1, device='cpu',
                 max_length=32768, use_bf16=False, **kwargs):
        super(Llama32, self).__init__(name=model_name, **kwargs)

        self.__max_length = max_length
        self.__pipe = pipeline("text-generation",
                               model=model_name,
                               torch_dtype=torch.bfloat16 if use_bf16 else "auto",
                               device_map=device,
                               temperature=temp,
                               pad_token_id=128001,
                               token=api_token)

    def ask(self, prompt):
        outputs = self.__pipe(
            [{"role": "user", "content": prompt}],
            max_new_tokens=self.__max_length,
        )
        return outputs[0]["generated_text"][-1]["content"]
