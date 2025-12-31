# This implementation has been tested for
# googletrans==4.0.2
import asyncio
import logging

from googletrans import Translator


class GoogleTranslateModel:

    def __init__(self, **kwargs):
        pass  # no persistent Translator instance anymore

    @staticmethod
    async def _translate_value_async(value, src, dest, sec_delay=1, attempts=10):

        logger = logging.getLogger()
        logger.setLevel(50)

        async with Translator() as translator:
            for i in range(attempts):
                try:
                    translated = await translator.translate(
                        value,
                        src=src,
                        dest=dest
                    )
                    return translated.text
                except Exception:
                    logger.info(
                        f"Unable to perform translation. Try {i + 1} out of {attempts}."
                    )
                    await asyncio.sleep(sec_delay)

        raise Exception("Can't translate")

    @staticmethod
    def translate_value(event_loop, value, src, dest, sec_delay=1, attempts=10):
        return event_loop.run_until_complete(
            GoogleTranslateModel._translate_value_async(
                value=value,
                src=src,
                dest=dest,
                sec_delay=sec_delay,
                attempts=attempts,
            )
        )

    def get_func(self, src, dest, **kwargs):
        # We do auto-import so we not depend on the actually installed library.
        # Translation of the list of data.
        # Returns the list of strings.
        return lambda str_list: [
            GoogleTranslateModel.translate_value(value=s, dest=dest, src=src, event_loop=kwargs.get("event_loop"))
            for s in str_list]
