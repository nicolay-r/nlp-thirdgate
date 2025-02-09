# nlp-thirdgate

<p align="center">
    <img src="logo.png"/>
</p>

A hub of [**third-party NLP providers**](#third-party-providers) and tutorials to help you instantly handle your [**data iterator**](#data-iterators) with [no-string dependency apps](#no-string-apps).

The purpose is of this project is to share **Third-party providers** that could be combined into a single [**pipeline**](#pipeline-formation).

# Third-Party Providers

* ### [LLM](llm)
  * **Mistral.AI** [[provider]](llm/mistralai_150.py) [[🤖 models]](https://docs.mistral.ai/getting-started/models/models_overview/)
  * **OpenRouter.AI** [[provider]](llm/open_router.py) [[🤖 models]](https://openrouter.ai/models)
  * **Replicate.IO** [[provider]](llm/replicate_104.py) [[🤖 models]](https://replicate.com/pricing#language-models)
  * **OpenAI** provider:
    * ChatGPT [[provider]](llm/openai_156.py)
    * Qwen-2.5-Max [[bash-script]](https://github.com/nicolay-r/nlp-thirdgate/blob/master/tutorials/llm_qwen_25_max_chat.sh)
    * o1 [[provider]](llm/openai_o1.py)
  * **Transformers**:
    * DeepSeek-R1-distill-7b [[📙 qwen-notebook]](tutorials/llm_deep_seek_7b_distill_qwen2.ipynb)
      [[📙 llama3-notebook]](tutorials/llm_deep_seek_7b_distill_llama3.ipynb)
    * LLaMA-3 [[provider]](llm/transformers_llama.py)
    * Qwen-2 [[provider]](llm/transformers_qwen2.py)
    * Microsoft-Phi-2 [[provider]](llm/transformers_microsoft_phi_2.py)
    * Mistral [[provider]](llm/transformers_mistral.py)
    * Gemma [[provider]](llm/transformers_gemma.py)
    * Flan-T5 [[provider]](llm/transformers_flan_t5.py)
    * DeciLM [[provider]](llm/transformers_decilm.py)
* ### [NER](ner)
    * **DeepPavlov** [[provider]](ner/dp_130.py) [[📙 notebook]](tutorials/ner_deeppavlov_130.ipynb)
    * **Flair** [[provider]](ner/flair_0151.py) [[bash-script]](tutorials/ner_flair_0151.sh) [[🤖 models]](https://huggingface.co/flair)
    * **Spacy** [[provider]](ner/spacy_383.py) [[bash-script]](tutorials/ner_spacy_383.sh) [[🤖 models]](https://huggingface.co/spacy)
* ### [Text-translation](text-translation)
    * **GoogleTranslator** [[provider]](text-translation/googletrans_310a.py) [[📙 notebook]](tutorials/translate_texts_with_spans_via_googletrans.ipynb)


# Data Iterators

In this project we consider that each provider represent a wrapper over third-party app to handle iterator of data.
We consider `dict` python type for representing each record of the data.

# Pipeline Formation

If you wish to use several [third-party providers](#third-party-providers) all together for a 
[data-iterators](#data-iterators), it is recommented to adopt [`AREkit` framework](https://github.com/nicolay-r/AREkit) as a no-string solution for deploying pipeline that support batching mode.

# No-string Application

* [bulk-chain](https://github.com/nicolay-r/bulk-chain) -- framework for reasoning over your tabular data rows with any provided LLM
* [bulk-ner](https://github.com/nicolay-r/bulk-ner) -- framework for a quick third-party models binding for entities extraction from cells of long tabular data
* [bulk-translate](https://github.com/nicolay-r/bulk-translate) --  framework for translation of a massive stream of texts with native support of pre-annotated fixed-spans that are invariant for translator.
* [AREkit pipelines](https://github.com/nicolay-r/AREkit) -- toolkit for handling your textual data iterators with various NLP providers
