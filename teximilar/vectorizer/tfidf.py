import math

from teximilar.tokenizer.pretrained_tokenizer import tokenize


def term_frequency(term: str, document: str, mode: str = 'log') -> float | int:
    term_count = document.count(term)
    match mode:
        case 'simple':
            return term_count
        case 'log':
            return math.log(term_count + 1)
        case _:
            all_terms = len(tokenize(document))
            return term_count / all_terms
