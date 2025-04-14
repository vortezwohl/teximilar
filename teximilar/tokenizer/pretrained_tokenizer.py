import os.path

from transformers import AutoTokenizer

from teximilar.util.text_hash import text_hash

PATH = os.path.dirname(os.path.abspath(__file__))
__cache = os.path.join(PATH, '.cache')
__tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1', cache_dir=__cache)

__text_to_tokens_cache = dict()


def tokenize(text: str) -> tuple[list[str], list[int]]:
    global __text_to_tokens_cache
    _hash = text_hash(text)
    if _hash in __text_to_tokens_cache.keys():
        return __text_to_tokens_cache.get(_hash)
    tokens = __tokenizer.tokenize(text)
    __text_to_tokens_cache[_hash] = (tokens, __tokenizer.convert_tokens_to_ids(tokens))
    return __text_to_tokens_cache[_hash]
