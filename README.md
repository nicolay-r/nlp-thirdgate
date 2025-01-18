# nlp-thirdgate

A hub of [**third-party NLP providers**](#third-party-providers) and tutorials to help you instantly handle your [**data iterator**](#data-iterators) with no-string dependencies.

The purpose is of this project is to share **Third-party providers** that could be combined into a single [**pipeline**](#pipeline-formation).

# Third-Party Providers

* ### [Text-translation](text-translation)
    * **GoogleTranslator** provider [[code]](text-translation/googletrans_310a.py) [[📙 notebook]](tutorials/translate_texts_with_spans_via_googletrans.ipynb)


# Data Iterators

In this project we consider that each provider represent a wrapper over third-party app to handle iterator of data.
We consider `dict` python type for representing each record of the data.

# Pipeline Formation

If you wish to use several [third-party providers](#third-party-providers) all together for a 
[data-iterators](#data-iterators), it is recommented to adopt [`AREkit` framework](https://github.com/nicolay-r/AREkit) as a no-string solution for deploying pipeline that support batching mode.

