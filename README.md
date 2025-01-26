# nlp-thirdgate

<p align="center">
    <img src="logo.png"/>
</p>

A hub of [**third-party NLP providers**](#third-party-providers) and tutorials to help you instantly handle your [**data iterator**](#data-iterators) with [no-string dependency apps](#no-string-apps).

The purpose is of this project is to share **Third-party providers** that could be combined into a single [**pipeline**](#pipeline-formation).

# Third-Party Providers

* ### [LLM](text-translation)
  * **OpenRouter.AI** provider [[code]](llm/open_router.py)
  * **Replicate.IO** provider [[code]](llm/replicate_104.py)
* ### [Text-translation](text-translation)
    * **GoogleTranslator** provider [[code]](text-translation/googletrans_310a.py) [[📙 notebook]](tutorials/translate_texts_with_spans_via_googletrans.ipynb)


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
