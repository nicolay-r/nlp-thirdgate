import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bulk_chain.core.llm_base import BaseLM


class Mistral(BaseLM):

    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.1", temp=0.1,
                 device='cpu', max_new_tokens=None, use_bf16=False, **kwargs):
        super(Mistral, self).__init__(name=model_name, **kwargs)

        self.__device = device
        self.__max_new_tokens = max_new_tokens
        self.__check_params()
        self.__model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16 if use_bf16 else "auto")
        self.__model.to(device)
        self.__tokenizer = AutoTokenizer.from_pretrained(
            model_name, torch_dtype=torch.bfloat16 if use_bf16 else "auto")
        self.__temp = temp

    def ask(self, prompt):
        inputs = self.__tokenizer(f"""<s>[INST]{prompt}[/INST]""", return_tensors="pt", add_special_tokens=False)
        inputs.to(self.__device)
        outputs = self.__model.generate(**inputs, max_new_tokens=self.__max, temperature=self.__temp,
                                        do_sample=True, pad_token_id=50256)
        response = self.__tokenizer.batch_decode(outputs)[0]
        parts = response.split("[/INST]")
        return "".join(parts[1:]) if len(parts) > 1 else ""
