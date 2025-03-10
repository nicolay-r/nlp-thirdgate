import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bulk_chain.core.llm_base import BaseLM


class Gemma(BaseLM):
    """ transformers==4.44.2
    """

    def __init__(self, model_name="google/gemma-7b-it", temp=0.1, device='cpu',
                 max_new_tokens=None, api_token=None, use_bf16=False, **kwargs):
        super(Gemma, self).__init__(name=model_name, support_batching=True, **kwargs)
        self.__device = device
        self.__model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16 if use_bf16 else "auto", token=api_token)
        self.__max_new_tokens = max_new_tokens
        self.__model.to(device)
        self.__tokenizer = AutoTokenizer.from_pretrained(model_name, token=api_token)
        self.__temp = temp

    def ask(self, batch):
        inputs = self.__tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        inputs.to(self.__device)
        outputs = self.__model.generate(**inputs, max_new_tokens=self.__max_new_tokens, temperature=self.__temp,
                                        do_sample=True)
        return self.__tokenizer.batch_decode(outputs, skip_special_tokens=True)
