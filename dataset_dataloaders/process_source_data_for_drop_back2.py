import collections
import copy
import json
import os
from typing import OrderedDict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from config.hparams import PARAMS
from tools.drop_utils import get_answer_type, AnswerAccessor, extract_answer_info_from_annotation, \
    word_tokenize, find_valid_spans, convert_word_to_number, find_valid_add_sub_expressions, \
    find_valid_counts, ALL_ANSWER_TYPES, convert_token_to_idx, find_valid_negations, \
    convert_answer_spans
from tools.log import get_logger


def get_examples(_hparams, _tokenizer, filePath):
    """
    :param _tokenizer:
    :param _hparams:
    :param filePath:
    :return: examples
    """
    _log = get_logger(log_name="DropDataloader")
    _examples = []
    eval_examples = {}
    # nlp = spacy.blank("en")

    total_items = 0
    _log.info('Load source data from ' + filePath)
    source_data = json.load(open(filePath, 'r'))
    _log.info('Prepare to process raw data')
    for passage_id, article in tqdm(source_data.items()):
        passage_text = article['passage'].replace("''", '" ').replace("``", '" ')
        passage_tokens = word_tokenize(sent=passage_text, tokenizer=_tokenizer)
        passage_spans = convert_token_to_idx(' '.join(passage_tokens), passage_tokens)

        for qa_pair in article['qa_pairs']:
            total_items += 1
            question_id = qa_pair['query_id']
            question_text = qa_pair['question'].replace("''", '" ').replace("``", '" ')
            question_tokens = word_tokenize(sent=question_text, tokenizer=_tokenizer
                                            )

            answer_annotations = list()
            answer_type = None
            if 'answer' in qa_pair and qa_pair['answer']:
                answer = qa_pair['answer']
                answer_type = get_answer_type(answer)
                if answer_type is None or answer_type not in ALL_ANSWER_TYPES:
                    continue
                answer_annotations.append(answer)

            if 'validated_answers' in qa_pair and qa_pair['validated_answers']:
                answer_annotations += qa_pair['validated_answers']

            if _hparams.do_train and answer_type is None:
                continue

            if answer_annotations:
                # answer type: "number" or "span". answer texts: number or list of spans
                # answer_annotations[0] 表示 answer
                answer_accessor, answer_texts = extract_answer_info_from_annotation(answer_annotations[0])
                if answer_accessor == AnswerAccessor.SPAN.value:
                    answer_texts = list(OrderedDict.fromkeys(answer_texts))
                # 控制答案长度
                number_of_answer = _hparams.max_answer_nums_length if len(
                    answer_texts) > _hparams.max_answer_nums_length else len(
                    answer_texts)

                # Tokenize and recompose the answer text in order to find the matching span based on token
                tokenized_answer_texts = []
                for answer_text in answer_texts:
                    answer_tokens = word_tokenize(sent=answer_text, tokenizer=_tokenizer)
                    tokenized_answer_texts.append(" ".join(answer_tokens))

                    # answer_tokens = word_tokenize(sent=answer_text, tokenizer=_tokenizer)
                    # answer_tokens = split_tokens_by_hyphen(answer_tokens)
                    # tokenized_answer_texts.append(answer_tokens)
                # log.info(tokenized_answer_texts)

                # 记录passage 中数字所在的位置
                numbers_in_passage = [0]
                number_indices = [-1]
                for token_index, token in enumerate(passage_tokens):
                    number = convert_word_to_number(token)
                    if number is not None:
                        numbers_in_passage.append(number)
                        number_indices.append(token_index)

                target_numbers = []
                # `answer_texts` is a list of valid answers.
                for answer_text in answer_texts:
                    number = convert_word_to_number(answer_text)
                    if number is not None:
                        target_numbers.append(number)

                valid_passage_spans = (
                    find_valid_spans(passage_tokens, tokenized_answer_texts)
                    if tokenized_answer_texts
                    else []
                )
                valid_question_spans = (
                    find_valid_spans(question_tokens, tokenized_answer_texts)
                    if tokenized_answer_texts
                    else []
                )

                # 提取出非阿拉伯数字位置 仅支持整数
                if not valid_passage_spans and answer_type in ["number", "date"]:
                    try:
                        if int(answer_texts[0]) in numbers_in_passage:
                            temp_index = numbers_in_passage.index(int(answer_texts[0]))
                            index_of_number = number_indices[temp_index]
                            valid_passage_spans = [(index_of_number, index_of_number)]
                    except ValueError:
                        pass

                # 如果没有从原文中提取出答案 则设置答案为None
                number_of_answer = None if valid_passage_spans == [] and valid_question_spans == [] else number_of_answer
                valid_signs_for_add_sub_expressions = []
                valid_counts = []
                if answer_type in ["number", "date"]:
                    valid_signs_for_add_sub_expressions = find_valid_add_sub_expressions(
                        numbers_in_passage, target_numbers,
                        max_number_of_numbers_to_consider=_hparams.max_number_of_numbers_to_consider
                    )

                if answer_type in ["number"]:
                    # Support count number 0 ~ max_count. Does not support float
                    numbers_for_count = list(range(_hparams.max_count))
                    valid_counts = find_valid_counts(numbers_for_count, target_numbers)  # valid indices
                    valid_counts = [count for count in valid_counts]

                valid_negations = find_valid_negations(numbers_in_passage, target_numbers)

                # Discard when no valid answer is available
                if valid_counts == [] and valid_passage_spans == [] and valid_question_spans == []:
                    continue

                example = {
                    "passage_id": passage_id,
                    "question_id": question_id,
                    "passage_tokens": passage_tokens,  # 分词后文本
                    "question_tokens": question_tokens,
                    "numbers_in_passage": numbers_in_passage,  # 文档中数字
                    "number_index": number_indices,  # 文档中数字所在下标
                    "answer_type": answer_type,
                    "number_of_answer": number_of_answer,  # 答案长度
                    "passage_spans": valid_passage_spans,  # 答案所在范围 tuple
                    "question_spans": valid_question_spans,
                    "add_sub_expressions": valid_signs_for_add_sub_expressions,
                    "counts": valid_counts,
                    "negations": valid_negations,
                    "answer_annotations": answer_annotations,
                }
                _examples.append(example)
                eval_examples[str(total_items)] = {
                    "context": passage_text,
                    "question": question_text,
                    "spans": passage_spans,
                    "answer": answer_annotations,
                    # answer: answer_annotations[0]  validated_answers: answer_annotations[1]
                    "uuid": passage_id  # for submission only
                }

    return _examples, eval_examples


def get_features(_hparams, _tokenizer, _evaluate=False, file_path=None):
    _log = get_logger(log_name="DropDataloader")
    _log.info('Load dataset with evaluate = ' + str(_evaluate))

    _examples, _eval_examples = get_examples(_hparams=_hparams, _tokenizer=_tokenizer,
                                             filePath=file_path)
    # TODO 构造输入
    skip_count, truncate_count = 0, 0
    unique_id = 1000000000
    _features = []
    for (example_index, example) in enumerate(_examples):
        question_index = []
        question_tokens = []
        for (i, token) in enumerate(example['question_tokens']):
            question_index.append(i)
            question_tokens.append(token)

        passage_index = []
        passage_tokens = []
        for (i, token) in enumerate(example['passage_tokens']):
            passage_index.append(i)
            passage_tokens.append(token)

        # 确保问题完整 截断 passage
        max_tokens_for_passage = _hparams.max_question_add_passage_length - len(question_tokens) - 3
        all_passage_tokens_len = len(passage_tokens)
        if all_passage_tokens_len > max_tokens_for_passage:
            passage_tokens = passage_tokens[:max_tokens_for_passage]
            truncate_count += 1
        # 截断 number_index = example.number_index
        number_index = []
        for index in example['number_index']:
            if index != -1:
                temp_index = passage_index[index]
                if temp_index < len(passage_tokens):
                    number_index.append(temp_index)
            else:
                number_index.append(-1)

        # 抽取 答案 span 下标
        query_tok_start_positions, query_tok_end_positions = \
            convert_answer_spans(example['question_spans'], question_index, len(question_tokens), question_tokens)

        passage_tok_start_positions, passage_tok_end_positions = \
            convert_answer_spans(example['passage_spans'], passage_index, all_passage_tokens_len, passage_tokens)

        #  [CLS] + [question token] + [SEP] + [passage token] + [SEP]
        # Truncate the passage according to the max sequence length
        input_tokens = []  # question 和 passage 拼接
        segment_ids = []  # 区分 question 和 passage
        que_token_to_orig_map = {}
        doc_token_to_orig_map = {}
        input_tokens.append('[CLS]')
        segment_ids.append(0)
        for i in range(len(question_tokens)):
            que_token_to_orig_map[len(input_tokens)] = question_index[i]
            input_tokens.append(question_tokens[i])
            segment_ids.append(0)
        input_tokens.append('[SEP]')
        segment_ids.append(0)
        # 拼接passage
        for i in range(len(passage_tokens)):
            doc_token_to_orig_map[len(input_tokens)] = passage_index[i]
            input_tokens.append(passage_tokens[i])
            segment_ids.append(1)
        input_tokens.append('[SEP]')
        segment_ids.append(1)
        # convert_tokens_to_ids
        input_ids = _tokenizer.convert_tokens_to_ids(input_tokens)
        # The mask has 1 for real tokens and 0 for padding tokens.
        input_mask = [1] * len(input_ids)

        # 更新 number_index
        input_number_index = []
        que_offset = 1  # +1 是因为添加了 [CLS]
        doc_offset = len(question_tokens) + 2  # +2 是因为添加了 [CLS]和[SEP]
        for temp_number_index in number_index:
            if temp_number_index != -1:
                temp_number_index = temp_number_index + doc_offset
                input_number_index.append(temp_number_index)
            else:
                input_number_index.append(-1)

        start_indices, end_indices, add_sub_expressions, input_counts, negations, number_of_answers = [], [], [], [], [], []
        if not _evaluate:
            # main
            # For distant supervision, we annotate the positions of all answer spans
            if passage_tok_start_positions != [] and passage_tok_end_positions != []:
                for tok_start_position, tok_end_position in zip(passage_tok_start_positions,
                                                                passage_tok_end_positions):
                    start_position = tok_start_position + doc_offset
                    end_position = tok_end_position + doc_offset
                    start_indices.append(start_position)
                    end_indices.append(end_position)
            elif query_tok_start_positions != [] and query_tok_end_positions != []:
                # 如果passage中没有抽取出span 从问题中抽取
                for tok_start_position, tok_end_position in zip(query_tok_start_positions, query_tok_end_positions):
                    start_position = tok_start_position + que_offset
                    end_position = tok_end_position + que_offset
                    start_indices.append(start_position)
                    end_indices.append(end_position)

            # Weakly-supervised for addition-subtraction
            if example['add_sub_expressions']:
                for add_sub_expression in example['add_sub_expressions']:
                    # 由于截断了 passage 导致 number_index 截断, 因此 expression 也应该被截断
                    if sum(add_sub_expression[:len(input_number_index)]) >= 2:
                        assert len(add_sub_expression[:len(input_number_index)]) == len(input_number_index)
                        add_sub_expressions.append(add_sub_expression[:len(input_number_index)])

            # Weakly-supervised for counting
            for count in example['counts']:
                input_counts.append(count)

            # Weakly-supervised for negation
            if example['negations']:
                for negation in example['negations']:
                    if sum(negation[:len(input_number_index)]) == 1:
                        assert len(negation[:len(input_number_index)]) == len(input_number_index)
                        negations.append(negation[:len(input_number_index)])

            is_impossible = True
            if "span_extraction" in _hparams.answering_abilities and start_indices != [] and end_indices != []:
                is_impossible = False
                assert example['number_of_answer'] is not None
                number_of_answers.append(example['number_of_answer'] - 1)

            if "negation" in _hparams.answering_abilities and negations != []:
                is_impossible = False

            if "addition_subtraction" in _hparams.answering_abilities and add_sub_expressions != []:
                is_impossible = False

            if "counting" in _hparams.answering_abilities and input_counts != []:
                is_impossible = False

            if start_indices == [] and end_indices == [] and number_of_answers == []:
                start_indices.append(-1)
                end_indices.append(-1)
                number_of_answers.append(-1)

            if not negations:
                negations.append([-1] * len(input_number_index))

            if not add_sub_expressions:
                add_sub_expressions.append([-1] * len(input_number_index))

            if not input_counts:
                input_counts.append(-1)

            if not is_impossible:
                feature = {
                    # "unique_id": example['question_id'],
                    "unique_id": unique_id,
                    "example_index": example_index,
                    "tokens": input_tokens,
                    "que_token_to_orig_map": que_token_to_orig_map,
                    "doc_token_to_orig_map": doc_token_to_orig_map,
                    "input_ids": input_ids,
                    "input_mask": input_mask,
                    "segment_ids": segment_ids,
                    "number_indices": number_index,
                    "start_indices": start_indices,
                    "end_indices": end_indices,
                    "number_of_answers": number_of_answers,
                    "add_sub_expressions": add_sub_expressions,
                    "input_counts": input_counts,
                    "negations": negations}
                _features.append(feature)
                unique_id += 1
            else:
                skip_count += 1

        else:
            feature = {
                # "unique_id": example['question_id'],
                "unique_id": unique_id,
                "example_index": example_index,
                "tokens": input_tokens,
                "que_token_to_orig_map": que_token_to_orig_map,
                "doc_token_to_orig_map": doc_token_to_orig_map,
                "input_ids": input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
                "number_indices": number_index}
            _features.append(feature)
            unique_id += 1
        if len(_features) % 5000 == 0:
            _log.info("Processing features: %d" % (len(_features)))
    _log.info(
        f"Skipped {skip_count} features, truncated {truncate_count} features, kept {len(_features)} features.")

    return _examples, _features


def get_tensors(feature, _hparams, is_train):
    start_indices = None
    end_indices = None
    number_of_answers = None
    input_counts = None
    new_add_sub_expressions = None
    new_negations = None

    # padding
    input_ids = copy.deepcopy(feature['input_ids'])
    input_mask = copy.deepcopy(feature['input_mask'])
    segment_ids = copy.deepcopy(feature['segment_ids'])
    # Zero-pad up to the max mini-batch sequence length.

    while len(input_ids) < _hparams.max_question_add_passage_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    number_indices = copy.deepcopy(feature['number_indices'])
    if number_indices is not None:
        number_indices = number_indices[0:_hparams.max_number_indices_len]
    while len(number_indices) < _hparams.max_number_indices_len:
        number_indices.append(-1)

    if is_train:
        start_indices = copy.deepcopy(feature['start_indices'])
        end_indices = copy.deepcopy(feature['end_indices'])
        number_of_answers = copy.deepcopy(feature['number_of_answers'])
        input_counts = copy.deepcopy(feature['input_counts'])
        add_sub_expressions = copy.deepcopy(feature['add_sub_expressions'])
        negations = copy.deepcopy(feature['negations'])

        if start_indices is not None:
            start_indices = start_indices[0:_hparams.max_answer_nums_length]
            end_indices = end_indices[0:_hparams.max_answer_nums_length]
        while len(start_indices) < _hparams.max_answer_nums_length:
            start_indices.append(-1)
            end_indices.append(-1)

        if input_counts is not None:
            input_counts = input_counts[0: _hparams.max_auxiliary_len]
        while len(input_counts) < _hparams.max_auxiliary_len:
            input_counts.append(-1)

        if number_of_answers is not None:
            number_of_answers = number_of_answers[0: _hparams.max_answer_nums_length]
        while len(number_of_answers) < _hparams.max_answer_nums_length:
            number_of_answers.append(-1)

        new_add_sub_expressions = []
        for add_sub_expression in add_sub_expressions:
            if add_sub_expression is not None:
                add_sub_expression = add_sub_expression[0: _hparams.max_number_indices_len]
            while len(add_sub_expression) < _hparams.max_number_indices_len:
                add_sub_expression.append(-1)
            new_add_sub_expressions.append(add_sub_expression)

        if new_add_sub_expressions is not None:
            new_add_sub_expressions = new_add_sub_expressions[0: _hparams.max_add_sub_combination_len]
        while len(new_add_sub_expressions) < _hparams.max_add_sub_combination_len:
            new_add_sub_expressions.append([-1] * _hparams.max_number_indices_len)

        new_negations = []
        for negation in negations:
            if negation is not None:
                negation = negation[0: _hparams.max_number_indices_len]
            while len(negation) < _hparams.max_number_indices_len:
                negation.append(-1)
            new_negations.append(negation)

        if new_negations is not None:
            new_negations = new_negations[0: _hparams.max_negation_combination_len]
        while len(new_negations) < _hparams.max_negation_combination_len:
            new_negations.append([-1] * _hparams.max_number_indices_len)

    # 转换成Tensor
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.bool)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    number_indices = torch.tensor(number_indices, dtype=torch.long)
    if is_train:
        start_indices = torch.tensor(start_indices, dtype=torch.long)
        end_indices = torch.tensor(end_indices, dtype=torch.long)
        number_of_answers = torch.tensor(number_of_answers, dtype=torch.long)
        input_counts = torch.tensor(input_counts, dtype=torch.long)
        add_sub_expressions = torch.tensor(new_add_sub_expressions, dtype=torch.long)
        negations = torch.tensor(new_negations, dtype=torch.long)

        return input_ids, input_mask, segment_ids, number_indices, start_indices, \
               end_indices, number_of_answers, input_counts, add_sub_expressions, negations
    else:
        return input_ids, input_mask, segment_ids, number_indices


def load_dataset(_hparams, _tokenizer, _evaluate=False, file_path=None):
    _log = get_logger(log_name="DropDataloader")
    _log.info('Load dataset with evaluate = ' + str(_evaluate))
    # 缓存
    from_train_or_test = file_path.split('/')[-1].split(".")[0]
    temp_file = "cache_from_{}_for_{}.pth".format(
        from_train_or_test,
        "test" if _evaluate else "train"
    )

    cache_file_path = os.path.join(_hparams.cachePath, temp_file)
    if os.path.exists(cache_file_path):
        # 加载缓存数据
        _log.info("Loading dataset from cached file %s", cache_file_path)
        dataset_and_examples = torch.load(cache_file_path)
        _dataset_tensors, _dataset_features, _examples = dataset_and_examples["dataset_tensors"], dataset_and_examples[
            "dataset_features"], dataset_and_examples["examples"]
        # 封装成 3元组
        datasets = []
        for tensor, feature in zip(_dataset_tensors, _dataset_features, ):
            datasets.append([tensor, feature])
        # return _dataset_tensors, _dataset_features, _examples
        return datasets,_examples

    else:
        _log.info("Can not load cache from " + cache_file_path)
        _log.info("Creating dataset ...")

        features_list, tensors_list = [], []

        _examples, _features = get_features(_hparams=_hparams, _tokenizer=_tokenizer, _evaluate=_evaluate,
                                            file_path=file_path)
        for feature in _features:
            tensors_list.append(get_tensors(feature=feature, _hparams=_hparams, is_train=not _evaluate))
            features_list.append(feature)

        _log.info('Prepared to save cache')
        torch.save({"dataset_tensors": tensors_list, "dataset_features": features_list, "examples": _examples},
                   cache_file_path)
        _log.info('Cache has been saved to ' + cache_file_path)
        # 封装成 3元组
        datasets = []
        for tensor, feature, example in zip(tensors_list, features_list, _examples):
            datasets.append([tensor, feature, example])
        # return tensors_list, features_list, _examples
        return datasets,_examples


if __name__ == '__main__':
    hparams = PARAMS
    hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)
    tokenizer = AutoTokenizer.from_pretrained(hparams.pretrainedModelPath)
    log = get_logger('process_source_data_for_drop')
    # get_examples(hparams=hparams_for_all, filePath=os.path.join(hparams_for_all.datasetPath, hparams_for_all.trainFile))
    dataset, _feature,examples = load_dataset(_hparams=hparams, _tokenizer=tokenizer, _evaluate=False,file_path=os.path.join(hparams.datasetPath, hparams.trainFile))
    # dataset = load_dataset(_hparams=hparams, _tokenizer=tokenizer, _evaluate=False,
    #                        file_path=os.path.join(hparams.datasetPath, hparams.trainFile))
    print(1)
