import itertools
import string
from collections import defaultdict
from enum import Enum
from typing import List

from word2number.w2n import word_to_num


class AnswerType(Enum):
    SINGLE_SPAN = 'single_span'
    MULTI_SPAN = 'multiple_span'
    NUMBER = 'number'
    DATE = 'date'


class AnswerAccessor(Enum):
    SPAN = 'spans'
    NUMBER = 'number'
    DATE = 'date'


WORD_NUMBER_MAP = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

SPAN_ANSWER_TYPES = [AnswerType.SINGLE_SPAN.value, AnswerType.MULTI_SPAN.value]
ALL_ANSWER_TYPES = SPAN_ANSWER_TYPES + [AnswerType.NUMBER.value, AnswerType.DATE.value]
STRIPPED_CHARACTERS = string.punctuation + "".join(["‘", "’", "´", "`", "_"])
IGNORED_TOKENS = {"a", "an", "the"}


def get_answer_type(answer):
    if answer['number']:
        return AnswerType.NUMBER.value
    elif answer['spans']:
        if len(answer['spans']) == 1:
            return AnswerType.SINGLE_SPAN.value
        return AnswerType.MULTI_SPAN.value
    elif any(answer['date'].values()):
        return AnswerType.DATE.value
    else:
        return None


def extract_answer_info_from_annotation(answer_annotation):
    answer_type = None
    if answer_annotation["spans"]:
        answer_type = "spans"
    elif answer_annotation["number"]:
        answer_type = "number"
    elif any(answer_annotation["date"].values()):
        answer_type = "date"

    answer_content = answer_annotation[answer_type] if answer_type is not None else None

    answer_texts: List[str] = []
    if answer_type is None:  # No answer
        pass
    elif answer_type == "spans":
        # answer_content is a list of string in this case
        answer_texts = answer_content
    elif answer_type == "date":
        # answer_content is a dict with "month", "day", "year" as the keys
        date_tokens = [
            answer_content[key]
            for key in ["month", "day", "year"]
            if key in answer_content and answer_content[key]
        ]
        answer_texts = date_tokens
    elif answer_type == "number":
        # answer_content is a string of number
        answer_texts = [answer_content]
    return answer_type, answer_texts


def word_tokenize(sent, tokenizer):
    # doc = nlp(sent)
    # return [token.text for token in doc]
    return tokenizer(sent, add_special_tokens=False).encodings[0].tokens


def tokenize_sentence(sent, tokenizer, max_length):
    doc = tokenizer(sent, max_length=max_length, padding='max_length', truncation=True)
    return doc.data['input_ids'], doc.data['attention_mask']


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print(f"Token {token} cannot be found")
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def convert_token_to_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print(f"Token {token} cannot be found")
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def convert_word_to_number(word: str, try_to_include_more_numbers=False):
    """
    Currently we only support limited types of conversion.
    """
    if try_to_include_more_numbers:
        # strip all punctuations from the sides of the word, except for the negative sign
        punctuations = string.punctuation.replace("-", "")
        word = word.strip(punctuations)
        # some words may contain the comma as deliminator
        word = word.replace(",", "")
        # word2num will convert hundred, thousand ... to number, but we skip it.
        if word in ["hundred", "thousand", "million", "billion", "trillion"]:
            return None
        try:
            number = word_to_num(word)
        except ValueError:
            try:
                number = int(word)
            except ValueError:
                try:
                    number = float(word)
                except ValueError:
                    number = None
        return number
    else:
        no_comma_word = word.replace(",", "")
        if no_comma_word in WORD_NUMBER_MAP:
            number = WORD_NUMBER_MAP[no_comma_word]
        else:
            try:
                number = int(no_comma_word)
            except ValueError:
                number = None
        return number


def find_valid_add_sub_expressions(
        numbers: List[int], targets: List[int], max_number_of_numbers_to_consider: int = 2
) -> List[List[int]]:
    valid_signs_for_add_sub_expressions = []
    for number_of_numbers_to_consider in range(2, max_number_of_numbers_to_consider + 1):
        possible_signs = list(itertools.product((-1, 1), repeat=number_of_numbers_to_consider))
        for number_combination in itertools.combinations(
                enumerate(numbers), number_of_numbers_to_consider
        ):
            indices = [it[0] for it in number_combination]
            values = [it[1] for it in number_combination]
            for signs in possible_signs:
                eval_value = sum(sign * value for sign, value in zip(signs, values))
                if eval_value in targets:
                    labels_for_numbers = [0] * len(numbers)  # 0 represents ``not included''.
                    for index, sign in zip(indices, signs):
                        labels_for_numbers[index] = (
                            1 if sign == 1 else 2
                        )  # 1 for positive, 2 for negative
                    valid_signs_for_add_sub_expressions.append(labels_for_numbers)
    return valid_signs_for_add_sub_expressions


def find_valid_spans(passage_tokens, answer_texts):
    # answer texts = tokenized and recomposed answer texts
    normalized_tokens = [
        token.lower().strip(STRIPPED_CHARACTERS) for token in passage_tokens
    ]
    word_positions = defaultdict(list)  # ?
    for i, token in enumerate(normalized_tokens):
        word_positions[token].append(i)  # dict telling index at which appears each word in the passage

    spans = []
    for answer_text in answer_texts:
        answer_tokens = answer_text.lower().strip(STRIPPED_CHARACTERS).split()
        num_answer_tokens = len(answer_tokens)
        if answer_tokens[0] not in word_positions:
            continue
        for span_start in word_positions[answer_tokens[0]]:
            span_end = span_start  # span_end is _inclusive_
            answer_index = 1
            while answer_index < num_answer_tokens and span_end + 1 < len(normalized_tokens):
                token = normalized_tokens[span_end + 1]
                if answer_tokens[answer_index].strip(STRIPPED_CHARACTERS) == token:
                    answer_index += 1
                    span_end += 1
                elif token in IGNORED_TOKENS:
                    span_end += 1
                else:
                    break
            if num_answer_tokens == answer_index:
                spans.append((span_start, span_end))
    return spans  # list of all matching passage slices


def find_valid_counts(count_numbers: List[int], targets: List[int]) -> List[int]:
    valid_indices = []
    for index, number in enumerate(count_numbers):
        if number in targets:
            valid_indices.append(index)
    return valid_indices
