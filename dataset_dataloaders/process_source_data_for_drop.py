import collections
import json
import os
from typing import OrderedDict

import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, is_torch_available

from config.hparams import PARAMS
from tools.drop_utils import get_answer_type, AnswerAccessor, extract_answer_info_from_annotation, \
    word_tokenize, find_valid_spans, convert_word_to_number, find_valid_add_sub_expressions, \
    find_valid_counts, ALL_ANSWER_TYPES, tokenize_sentence, convert_token_to_idx
from tools.log import get_logger
from tools.tool import save_json_data_to_file


def get_examples(_hparams, _tokenizer, filePath):
    """
    :param _tokenizer:
    :param _hparams:
    :param filePath:
    :return: examples
    """
    _examples = []
    eval_examples = {}
    # nlp = spacy.blank("en")
    log = get_logger(log_name="get_examples")
    total_items = 0
    log.info('Load source data from ' + filePath)
    source_data = json.load(open(filePath, 'r'))
    log.info('Prepare to process raw data')
    for passage_id, article in tqdm(source_data.items()):
        passage_text = article['passage'].replace("''", '" ').replace("``", '" ')
        passage_tokens = word_tokenize(sent=passage_text, tokenizer=_tokenizer)
        passage_spans = convert_token_to_idx(' '.join(passage_tokens), passage_tokens)

        for qa_pair in article['qa_pairs']:
            total_items += 1
            question_id = qa_pair['query_id']
            question_text = qa_pair['question'].replace("''", '" ').replace("``", '" ')
            question_tokens = word_tokenize(sent=question_text, tokenizer=_tokenizer)

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
                # Tokenize and recompose the answer text in order to find the matching span based on token
                tokenized_answer_texts = []
                for answer_text in answer_texts:
                    answer_tokens = word_tokenize(sent=answer_text, tokenizer=_tokenizer)
                    tokenized_answer_texts.append(" ".join(token for token in answer_tokens))
                # log.info(tokenized_answer_texts)

                # 记录passage 中数字所在的位置
                numbers_in_passage = []
                number_indices = []
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
                # 提取出非阿拉伯数字位置 仅支持整数
                if not valid_passage_spans and answer_type in ["number", "date"]:
                    try:
                        if int(answer_texts[0]) in numbers_in_passage:
                            temp_index = numbers_in_passage.index(int(answer_texts[0]))
                            index_of_number = number_indices[temp_index]
                            valid_passage_spans = [(index_of_number, index_of_number)]
                    except ValueError:
                        pass

                valid_signs_for_add_sub_expressions = []
                valid_counts = []
                if answer_type in ["number", "date"]:
                    valid_signs_for_add_sub_expressions = find_valid_add_sub_expressions(
                        numbers_in_passage, target_numbers
                    )

                if answer_type in ["number"]:
                    # Support count number 0 ~ max_count. Does not support float
                    numbers_for_count = list(range(_hparams.max_count))
                    valid_counts = find_valid_counts(numbers_for_count, target_numbers)  # valid indices
                    valid_counts = [str(count) for count in valid_counts]

                # Discard when no valid answer is available
                if valid_counts == [] and valid_passage_spans == []:
                    continue

                # -1 if no answer is provided
                if not valid_passage_spans:
                    valid_passage_spans.append((-1, -1))
                if not valid_signs_for_add_sub_expressions:
                    valid_signs_for_add_sub_expressions.append([-1])
                if not valid_counts:
                    # 转换成字符串类型 防止tokenizer 报错
                    valid_counts.append('-1')
                if not number_indices:
                    number_indices.append('-1')

                # split start and end indices
                start_indices = []
                end_indices = []
                for span in valid_passage_spans:
                    start_indices.append(str(span[0]))
                    end_indices.append(str(span[1]))

                example = {
                    "passage_id": passage_id,
                    "passage_text": passage_text,  # 原始文本
                    "passage_tokens": passage_tokens,  # 分词后文本
                    "question_id": question_id,
                    "question_text": question_text,
                    "question_tokens": question_tokens,
                    "answer_type": answer_type,
                    "answer_annotations": answer_annotations,
                    # answer: answer_annotations[0]  validated_answers: answer_annotations[1]
                    "answer_texts": answer_texts,  # 答案 list
                    "valid_passage_spans": valid_passage_spans,  # 答案所在范围 tuple
                    "start_indices": ' '.join(start_indices),
                    "end_indices": ' '.join(end_indices),
                    "counts": ' '.join(valid_counts),
                    "id": str(total_items),  # 当前item索引
                    "number_index": number_indices,  # 文档中数字所在下标
                    "add_sub_expressions": valid_signs_for_add_sub_expressions
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


def load_dataset(_hparams, _tokenizer, _evaluate=False):
    """
    :param _hparams:
    :param _tokenizer:
    :param _evaluate:
    :return: _dataset, _examples
    """
    log = get_logger(log_name="load_dataset")
    log.info('Load dataset with evaluate = ' + str(_evaluate))
    # 缓存
    temp_file = "cache_from_{}_for_{}.pth".format(
        (_hparams.testFile if _evaluate else _hparams.trainFile).split(".")[0],
        "dev" if _evaluate else "train"
    )
    cache_file_path = os.path.join(_hparams.cachePath, temp_file)
    if os.path.exists(cache_file_path):
        # 加载缓存数据
        log.info("Loading dataset from cached file %s", cache_file_path)
        dataset_and_examples = torch.load(cache_file_path)
        _dataset, _examples = dataset_and_examples["dataset"], dataset_and_examples["examples"]
        return _dataset, _examples
    else:
        log.info("Can not load cache from " + cache_file_path)
        log.info("Creating dataset ...")
        if _evaluate:
            _examples, dev_eval = get_examples(_hparams=_hparams, _tokenizer=_tokenizer,
                                               filePath=os.path.join(_hparams.datasetPath, _hparams.testFile))
            save_json_data_to_file(filename=_hparams.dev_eval_file, json_data=dev_eval, message="dev eval", log=log)
        else:
            _examples, train_eval = get_examples(_hparams=_hparams, _tokenizer=_tokenizer,
                                                 filePath=os.path.join(_hparams.datasetPath, _hparams.trainFile))
            save_json_data_to_file(filename=_hparams.train_eval_file, json_data=train_eval, message="train eval",
                                   log=log)

        passage_text_input_ids, passage_text_attention_mask = [], []
        question_text_input_ids, question_text_attention_mask = [], []
        start_indices_input_ids, start_indices_attention_mask = [], []
        end_indices_input_ids, end_indices_attention_mask = [], []
        counts_input_ids, count_attention_mask = [], []
        id_input_ids, id_attention_mask = [], []

        log.info('Prepared to create tensor dataset')
        for example in _examples:
            temp_passage_text_input_ids, temp_passage_text_attention_mask = tokenize_sentence(
                sent=example['passage_text'],
                tokenizer=_tokenizer, max_length=_hparams.max_passage_length)
            temp_question_text_input_ids, temp_question_text_attention_mask = tokenize_sentence(
                sent=example['question_text'],
                tokenizer=_tokenizer, max_length=_hparams.max_question_length)
            temp_id_input_ids, temp_id_attention_mask = tokenize_sentence(
                sent=example['id'],
                tokenizer=_tokenizer, max_length=_hparams.max_auxiliary_len)
            if _evaluate:
                temp_start_indices_input_ids = [_tokenizer.cls_token_id] + [_tokenizer.pad_token_id] * (
                        _hparams.max_answer_span_length - 1)
                temp_start_indices_attention_mask = []

                temp_end_indices_input_ids = [_tokenizer.cls_token_id] + [_tokenizer.pad_token_id] * (
                        _hparams.max_answer_span_length - 1)
                temp_end_indices_attention_mask = []

                temp_counts_input_ids = [_tokenizer.cls_token_id] + [_tokenizer.pad_token_id] * (
                        _hparams.max_auxiliary_len - 1)
                temp_count_attention_mask = []
            else:
                temp_start_indices_input_ids, temp_start_indices_attention_mask = tokenize_sentence(
                    sent=example['start_indices'],
                    tokenizer=_tokenizer, max_length=_hparams.max_answer_span_length)
                temp_end_indices_input_ids, temp_end_indices_attention_mask = tokenize_sentence(
                    sent=example['end_indices'],
                    tokenizer=_tokenizer, max_length=_hparams.max_answer_span_length)
                temp_counts_input_ids, temp_count_attention_mask = tokenize_sentence(
                    sent=example['counts'],
                    tokenizer=_tokenizer, max_length=_hparams.max_auxiliary_len)

            passage_text_input_ids.append(temp_passage_text_input_ids)
            passage_text_attention_mask.append(temp_passage_text_attention_mask)
            question_text_input_ids.append(temp_question_text_input_ids)
            question_text_attention_mask.append(temp_question_text_attention_mask)
            start_indices_input_ids.append(temp_start_indices_input_ids)
            start_indices_attention_mask.append(temp_start_indices_attention_mask)
            end_indices_input_ids.append(temp_end_indices_input_ids)
            end_indices_attention_mask.append(temp_end_indices_attention_mask)
            counts_input_ids.append(temp_counts_input_ids)
            count_attention_mask.append(temp_count_attention_mask)
            id_input_ids.append(temp_id_input_ids)
            id_attention_mask.append(temp_id_attention_mask)

        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")
        log.info("Created dataset length = %d.", len(_examples))
        if _evaluate:
            _dataset = TensorDataset(
                torch.tensor(passage_text_input_ids, dtype=torch.long),
                torch.tensor(passage_text_attention_mask, dtype=torch.long),
                torch.tensor(question_text_input_ids, dtype=torch.long),
                torch.tensor(question_text_attention_mask, dtype=torch.long),
                torch.tensor(start_indices_input_ids, dtype=torch.long),
                torch.tensor(end_indices_input_ids, dtype=torch.long),
                torch.tensor(counts_input_ids, dtype=torch.long),
                torch.tensor(id_input_ids, dtype=torch.long),
                torch.tensor(id_attention_mask, dtype=torch.long)
            )
        else:
            _dataset = TensorDataset(
                torch.tensor(passage_text_input_ids, dtype=torch.long),
                torch.tensor(passage_text_attention_mask, dtype=torch.long),
                torch.tensor(question_text_input_ids, dtype=torch.long),
                torch.tensor(question_text_attention_mask, dtype=torch.long),
                torch.tensor(start_indices_input_ids, dtype=torch.long),
                torch.tensor(start_indices_attention_mask, dtype=torch.long),
                torch.tensor(end_indices_input_ids, dtype=torch.long),
                torch.tensor(end_indices_attention_mask, dtype=torch.long),
                torch.tensor(counts_input_ids, dtype=torch.long),
                torch.tensor(count_attention_mask, dtype=torch.long),
                torch.tensor(id_input_ids, dtype=torch.long),
                torch.tensor(id_attention_mask, dtype=torch.long)

            )
    log.info('Prepared to save cache')
    torch.save({"dataset": _dataset, "examples": _examples}, cache_file_path)
    log.info('Cache has been saved to ' + cache_file_path)
    return _dataset, _examples


if __name__ == '__main__':
    hparams = PARAMS
    hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)
    tokenizer = AutoTokenizer.from_pretrained(hparams.pretrainedModelPath)

    # get_examples(hparams=hparams_for_all, filePath=os.path.join(hparams_for_all.datasetPath, hparams_for_all.trainFile))
    dataset, examples = load_dataset(_hparams=hparams, _tokenizer=tokenizer, _evaluate=False)
