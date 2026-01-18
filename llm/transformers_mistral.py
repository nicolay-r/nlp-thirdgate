import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bulk_chain.core.llm_base import BaseLM


class Mistral(BaseLM):
    """ Tested under:
            transformers==4.44.2
            bulk-chain==1.2.1
    """

    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.1", temp=0.1,
                 device='cpu', max_new_tokens=None, use_bf16=False, **kwargs):
        super(Mistral, self).__init__(name=model_name, support_batching=True, **kwargs)

        self.__device = device
        self.__max_new_tokens = max_new_tokens
        self.__check_params()
        self.__model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16 if use_bf16 else "auto")
        self.__model.to(device)
        self.__tokenizer = AutoTokenizer.from_pretrained(
            model_name, torch_dtype=torch.bfloat16 if use_bf16 else "auto")
        self.__temp = temp

    @staticmethod
    def __handle(response):
        parts = response.split("[/INST]")
        return "".join(parts[1:]) if len(parts) > 1 else ""

    def ask_batch(self, batch):

        batch = [f"Instruct: {text}\nOutput:" for text in batch]

        inputs = self.__tokenizer(batch, return_tensors="pt")
        inputs.to(self.__device)
        outputs = self.__model.generate(**inputs, max_new_tokens=self.__max, temperature=self.__temp, do_sample=True)
        decoded_output = self.__tokenizer.batch_decode(outputs)

        return [self.__handle(response) for response in decoded_output]