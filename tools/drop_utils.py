import collections
import itertools
import string
from collections import Counter
from collections import defaultdict
from decimal import Decimal
from enum import Enum
from random import choice
from typing import List

import numpy as np
import six
from transformers.data.metrics.squad_metrics import _get_best_indexes, normalize_answer, _compute_softmax
from word2number.w2n import word_to_num

from drop.beam_search import beam_search
from drop.drop_eval import answer_json_to_strings, get_metrics
from tools import tokenization


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

sign_remap = {0: 0, 1: 1, 2: -1}
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
    return \
        tokenizer(sent, add_special_tokens=False).encodings[
            0].tokens


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


def find_valid_negations(numbers: List[int], targets: List[int]) -> List[List[int]]:
    valid_negations = []
    decimal_targets = [Decimal(x).quantize(Decimal('0.00')) for x in targets]
    for index, number in enumerate(numbers):
        decimal_negating_number = Decimal(100 - number).quantize(Decimal('0.00'))
        if 0 < number < 100 and decimal_negating_number in decimal_targets:
            labels_for_numbers = [0] * len(numbers)
            labels_for_numbers[index] = 1
            valid_negations.append(labels_for_numbers)
    return valid_negations


def convert_answer_spans(spans, orig_to_tok_index, all_len, all_tokens):
    tok_start_positions, tok_end_positions = [], []
    for span in spans:
        start_position, end_position = span[0], span[1]
        tok_start_position = orig_to_tok_index[start_position]
        if end_position + 1 >= len(orig_to_tok_index):
            tok_end_position = all_len - 1
        else:
            tok_end_position = orig_to_tok_index[end_position + 1] - 1
        if tok_start_position < len(all_tokens) and tok_end_position < len(all_tokens):
            tok_start_positions.append(tok_start_position)
            tok_end_positions.append(tok_end_position)
    return tok_start_positions, tok_end_positions


def batch_annotate_candidates(all_examples, batch_features, batch_results, answering_abilities,
                              is_training, beam_size, max_count):
    """Annotate top-k candidate answers into features."""
    unique_id_to_result = {}
    for result in batch_results:
        unique_id_to_result[result['unique_id']] = result

    batch_number_indices, batch_sign_indices, batch_sign_labels, batch_scores = [], [], [], []
    for (feature_index, feature) in enumerate(batch_features):
        example = all_examples[feature.example_index]
        result = unique_id_to_result[feature.unique_id]

        number_indices, sign_indices, sign_labels, scores = None, None, None, None
        if is_training:
            if feature.add_sub_expressions != [[-1] * len(feature.number_indices)]:
                number_indices, sign_indices, sign_labels, scores = add_sub_beam_search(example, feature, result,
                                                                                        is_training, beam_size,
                                                                                        max_count)
        else:
            predicted_ability = result['predicted_ability']
            predicted_ability_str = answering_abilities[predicted_ability]
            if predicted_ability_str == "addition_subtraction":
                number_indices, sign_indices, sign_labels, scores = add_sub_beam_search(example, feature, result,
                                                                                        is_training, beam_size,
                                                                                        max_count)

        if number_indices is None and sign_indices is None and sign_labels is None and scores is None:
            number_indices, sign_indices, sign_labels, scores = [], [], [], []
            while len(number_indices) < beam_size:
                number_indices.append([-1] * max_count)
                sign_indices.append([-1] * max_count)
                sign_labels.append(0)
                scores.append(0)

        batch_number_indices.append(number_indices)
        batch_sign_indices.append(sign_indices)
        batch_sign_labels.append(sign_labels)
        batch_scores.append(scores)
    return batch_number_indices, batch_sign_indices, batch_sign_labels, batch_scores


def add_sub_beam_search(example, feature, result, is_training, beam_size, max_count):
    number_sign_logits = result['number_sign_logits']  # [L, 3]
    number_mask = result['number_mask']  # [L]
    number_indices_list, sign_indices_list, scores_list = beam_search(number_sign_logits, number_mask, beam_size,
                                                                      max_count)

    number_sign_labels = []
    if is_training:
        if number_indices_list != [] and sign_indices_list != []:
            for number_indices, sign_indices in zip(number_indices_list, sign_indices_list):
                pred_answer = sum([example.numbers_in_passage[number_index] * sign_remap[sign_index]
                                   for number_index, sign_index in zip(number_indices, sign_indices)])
                pred_answer = float(Decimal(pred_answer).quantize(Decimal('0.0000')))
                ground_truth_answer_strings = [answer_json_to_strings(annotation)[0] for annotation in
                                               example.answer_annotations]
                exact_match, _ = metric_max_over_ground_truths(
                    get_metrics, str(pred_answer), ground_truth_answer_strings)
                number_sign_labels.append(exact_match)

    # Pad to fixed length
    for number_indices, sign_indices in zip(number_indices_list, sign_indices_list):
        while len(number_indices) < max_count:
            number_indices.append(-1)
            sign_indices.append(-1)

    while len(number_indices_list) < beam_size:
        number_indices_list.append([-1] * max_count)
        sign_indices_list.append([-1] * max_count)
        scores_list.append(0)
        if is_training:
            number_sign_labels.append(0)

    # Add ground truth expressions if there is no positive label
    if is_training and max(number_sign_labels) == 0:
        gold_number_indices, gold_sign_indices = [], []
        add_sub_expression = choice(feature.add_sub_expressions)
        for number_index, sign_index in enumerate(add_sub_expression):
            if sign_index > 0 and number_mask[number_index]:
                gold_number_indices.append(number_index)
                gold_sign_indices.append(sign_index)
        while len(gold_number_indices) < max_count:
            gold_number_indices.append(-1)
            gold_sign_indices.append(-1)
        number_indices_list[-1] = gold_number_indices
        sign_indices_list[-1] = gold_sign_indices
        number_sign_labels[-1] = 1

    return number_indices_list, sign_indices_list, number_sign_labels, scores_list


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def write_predictions(all_examples, all_features, all_results, answering_abilities, drop_metrics, length_heuristic,
                      n_best_size, max_answer_length, do_lower_case, verbose_logging, logger):
    """Write final predictions to the json file."""

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result['unique_id']] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["start_index", "end_index", "start_logit", "end_logit", "rerank_logit", "heuristic_logit"])

    all_nbest_json = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        assert len(features) == 1

        feature = features[0]
        result = unique_id_to_result[feature.unique_id]
        predicted_ability = result['predicted_ability']
        predicted_ability_str = answering_abilities[predicted_ability]
        nbest_json, predicted_answers = [], []
        if predicted_ability_str == "addition_subtraction":
            max_prob, best_answer = 0, None
            sign_rerank_probs = _compute_softmax(result['sign_rerank_logits'])
            for number_indices, sign_indices, rerank_prob, prob in zip(result['number_indices2'],
                                                                       result['sign_indices'], sign_rerank_probs,
                                                                       result['sign_probs']):
                pred_answer = sum(
                    [sign_remap[sign_index] * example.numbers_in_passage[number_index] for sign_index, number_index in
                     zip(sign_indices, number_indices) if sign_index != -1 and number_index != -1])
                pred_answer = str(float(Decimal(pred_answer).quantize(Decimal('0.0000'))))
                if rerank_prob * prob > max_prob:
                    max_prob = rerank_prob * prob
                    best_answer = pred_answer
            assert best_answer is not None
            predicted_answers.append(best_answer)
            output = collections.OrderedDict()
            output["text"] = str(best_answer)
            output["type"] = "addition_subtraction"
            nbest_json.append(output)
        elif predicted_ability_str == "counting":
            predicted_answers.append(str(result['predicted_count']))
            output = collections.OrderedDict()
            output["text"] = str(result['predicted_count'])
            output["type"] = "counting"
            nbest_json.append(output)
        elif predicted_ability_str == "negation":
            index = np.argmax(result['predicted_negations'])
            pred_answer = 100 - example.numbers_in_passage[index]
            pred_answer = float(Decimal(pred_answer).quantize(Decimal('0.0000')))
            predicted_answers.append(str(pred_answer))
            output = collections.OrderedDict()
            output["text"] = str(pred_answer)
            output["type"] = "negation"
            nbest_json.append(output)
        elif predicted_ability_str == "span_extraction":
            number_of_spans = result['predicted_spans']
            prelim_predictions = []
            start_indexes = _get_best_indexes(result['start_logits'], n_best_size)
            end_indexes = _get_best_indexes(result['end_logits'], n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.que_token_to_orig_map and start_index not in feature.doc_token_to_orig_map:
                        continue
                    if end_index not in feature.que_token_to_orig_map and start_index not in feature.doc_token_to_orig_map:
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    start_logit = result['start_logits'][start_index]
                    end_logit = result['end_logits'][end_index]
                    heuristic_logit = start_logit + end_logit \
                                      - length_heuristic * (end_index - start_index + 1)
                    prelim_predictions.append(
                        _PrelimPrediction(
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=start_logit,
                            end_logit=end_logit,
                            rerank_logit=0,
                            heuristic_logit=heuristic_logit))

            prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.heuristic_logit), reverse=True)

            _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                "NbestPrediction",
                ["text", "start_logit", "end_logit", "start_index", "end_index", "rerank_logit", "heuristic_logit"])

            seen_predictions = {}
            nbest = []
            for i, pred_i in enumerate(prelim_predictions):
                if len(nbest) >= n_best_size:
                    break

                final_text = wrapped_get_final_text(example, feature, pred_i.start_index, pred_i.end_index,
                                                    do_lower_case, verbose_logging, logger)
                if final_text in seen_predictions or final_text is None:
                    continue

                seen_predictions[final_text] = True
                nbest.append(
                    _NbestPrediction(
                        text=final_text,
                        start_logit=pred_i.start_logit,
                        end_logit=pred_i.end_logit,
                        start_index=pred_i.start_index,
                        end_index=pred_i.end_index,
                        rerank_logit=pred_i.rerank_logit,
                        heuristic_logit=pred_i.heuristic_logit
                    ))

                # filter out redundant candidates
                if (i + 1) < len(prelim_predictions):
                    indexes = []
                    for j, pred_j in enumerate(prelim_predictions[(i + 1):]):
                        filter_text = wrapped_get_final_text(example, feature, pred_j.start_index, pred_j.end_index,
                                                             do_lower_case, verbose_logging, logger)
                        if filter_text is None:
                            indexes.append(i + j + 1)
                        else:
                            if calculate_f1(final_text, filter_text) > 0:
                                indexes.append(i + j + 1)
                    [prelim_predictions.pop(index - k) for k, index in enumerate(indexes)]

            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(
                    _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start_index=0.0, end_index=0.0,
                                     rerank_logit=0., heuristic_logit=0.))

            assert len(nbest) >= 1

            for i, entry in enumerate(nbest):
                if i > number_of_spans:
                    break
                predicted_answers.append(entry.text)
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["type"] = "span_extraction"
                nbest_json.append(output)
        else:
            raise ValueError(f"Unsupported answer ability: {predicted_ability_str}")

        assert len(nbest_json) >= 1 and len(predicted_answers) >= 1
        if example.answer_annotations:
            drop_metrics(predicted_answers, example.answer_annotations)
        all_nbest_json[example.qas_id] = nbest_json

    exact_match, f1_score = drop_metrics.get_metric(reset=True)
    return all_nbest_json, {'em': exact_match, 'f1': f1_score}


def wrapped_get_final_text(example, feature, start_index, end_index, do_lower_case, verbose_logging, logger):
    if start_index in feature.doc_token_to_orig_map and end_index in feature.doc_token_to_orig_map:
        orig_doc_start = feature.doc_token_to_orig_map[start_index]
        orig_doc_end = feature.doc_token_to_orig_map[end_index]
        orig_tokens = example.passage_tokens[orig_doc_start:(orig_doc_end + 1)]
    elif start_index in feature.que_token_to_orig_map and end_index in feature.que_token_to_orig_map:
        orig_que_start = feature.que_token_to_orig_map[start_index]
        orig_que_end = feature.que_token_to_orig_map[end_index]
        orig_tokens = example.question_tokens[orig_que_start:(orig_que_end + 1)]
    else:
        return None

    tok_tokens = feature.tokens[start_index:(end_index + 1)]
    tok_text = " ".join(tok_tokens)

    # De-tokenize WordPieces that have been split off.
    tok_text = tok_text.replace(" ##", "")
    tok_text = tok_text.replace("##", "")

    # Clean whitespace
    tok_text = tok_text.strip()
    tok_text = " ".join(tok_text.split())
    orig_text = " ".join(orig_tokens)

    final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging, logger)
    return final_text


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False, logger=None):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return ns_text, ns_to_s_map

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def calculate_f1(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
