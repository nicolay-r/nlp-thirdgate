# nlp-thirdgate

A hub of **third-party providers** and tutorials to help you instantly apply various NLP techniques in Python towards your **data iterator** ([more](#data-iterators)) with minimum dependencies.

The purpose is to share **tiny** examples of using each provider that would allow you to combine them in a single **NLP pipeline** ([more](#pipeline-formation)) using your own frameworks.

# Third-Party Providers

* ### [Text-translation](text-translation)
    * **GoogleTranslator** provider [[code]](text-translation/googletrans_310a.py) [[📙 notebook]](tutorials/translate_texts_with_spans_via_googletrans.ipynb)


# Data Iterators

In this project we consider that each provider represent a wrapper over third-party app to handle iterator of data.
We consider `dict` python type for representing each record of the data.

# Pipeline Formation

If you wish to use several [third-party providers](#third-party-providers) all together for a 
[data-iterators](#data-iterators), it is recommented to adopt [`AREkit` framework](https://github.com/nicolay-r/AREkit) as a no-string solution for deploying pipeline that support batching mode.

