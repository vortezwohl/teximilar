import os.path

from transformers import AutoTokenizer

PATH = os.path.dirname(os.path.abspath(__file__))
__cache = os.path.join(PATH, '.cache')
__tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1', cache_dir=__cache)


def tokenize(text: str) -> list[int]:
    return __tokenizer.convert_tokens_to_ids(__tokenizer.tokenize(text))
