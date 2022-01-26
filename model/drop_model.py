import math

import torch
from pytorch_pretrained_bert import BertModel
from torch import nn
from torch.nn import functional
from torch.nn.functional import log_softmax


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def get_self_att_representation(input, input_score, input_mask, dim=1):
    '''
    :param input: [N, L, D]
    :param input_score: [N, L]
    :param input_mask: [N, L]
    :return: [N, D]
    '''
    input_mask = input_mask.to(dtype=input_score.dtype)  # fp16 compatibility
    input_mask = (1.0 - input_mask) * -10000.0
    input_score = input_score + input_mask
    input_prob = nn.Softmax(dim=-1)(input_score)
    input_prob = input_prob.unsqueeze(-1)
    output = torch.sum(input_prob * input, dim=dim)
    return output


def gather_representations(output, indices):
    _ndims = len(indices.size())
    if _ndims == 1:
        indices = indices.unsqueeze(-1)
    index_mask = (indices != -1).long()
    clamped_indices = replace_masked_values(indices, index_mask, 0)
    # Shape: (batch_size, # of indices, hidden_size)
    gathered_output = torch.gather(output, 1,
                                   clamped_indices.unsqueeze(-1).expand(-1, -1, output.size(-1)))
    return gathered_output


def replace_masked_values(tensor: torch.Tensor, mask: torch.Tensor, replace_with: float) -> torch.Tensor:
    """
    Replaces all masked values in ``tensor`` with ``replace_with``.  ``mask`` must be broadcastable
    to the same shape as ``tensor``. We require that ``tensor.dim() == mask.dim()``, as otherwise we
    won't know which dimensions of the mask to unsqueeze.

    This just does ``tensor.masked_fill()``, except the pytorch method fills in things with a mask
    value of 1, where we want the opposite.  You can do this in your own code with
    ``tensor.masked_fill((1 - mask).byte(), replace_with)``.
    """
    if tensor.dim() != mask.dim():
        raise ConfigurationError("tensor.dim() (%d) != mask.dim() (%d)" % (tensor.dim(), mask.dim()))
    return tensor.masked_fill((1 - mask).bool(), replace_with)


def gather_log_likelihood(log_probs, labels):
    gold_mask = (labels != -1).long()
    clamped_labels = replace_masked_values(labels, gold_mask, 0)

    # (batch_size, # of gold indices)
    log_likelihood = torch.gather(log_probs, 1, clamped_labels)
    return log_likelihood


def masked_logsumexp(log_likelihood, labels):
    gold_mask = (labels != -1).long()
    log_likelihood = replace_masked_values(log_likelihood, gold_mask, -1e7)
    log_marginal_likelihood = logsumexp(log_likelihood)
    return log_marginal_likelihood


def logsumexp(tensor: torch.Tensor,
              dim: int = -1,
              keep_dim: bool = False) -> torch.Tensor:
    """
    A numerically stable computation of logsumexp. This is mathematically equivalent to
    `tensor.exp().sum(dim, keep=keepdim).log()`.  This function is typically used for summing log
    probabilities.

    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A tensor of arbitrary size.
    dim : int, optional (default = -1)
        The dimension of the tensor to apply the logsumexp to.
    keep_dim: bool, optional (default = False)
        Whether to retain a dimension of size one at the dimension we reduce over.
    """
    max_score, _ = tensor.max(dim, keepdim=keep_dim)
    if keep_dim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keep_dim)).log()


def distant_cross_entropy(logits, labels):
    '''
    :param logits: [N, L]
    :param labels: [N, L]
    '''
    _log_softmax = nn.LogSoftmax(dim=-1)
    log_likelihood = _log_softmax(logits)

    log_likelihood = replace_masked_values(log_likelihood, labels, -1e7)
    log_marginal_likelihood = logsumexp(log_likelihood)
    return log_marginal_likelihood


class ConfigurationError(Exception):
    """
    The exception raised by any AllenNLP object when it's misconfigured
    (e.g. missing properties, invalid properties, unknown properties).
    """

    def __init__(self, message):
        super(ConfigurationError, self).__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)


class BertFeedForward(nn.Module):
    def __init__(self, config, input_size, intermediate_size, output_size):
        super(BertFeedForward, self).__init__()
        self.dense = nn.Linear(input_size, intermediate_size)
        self.affine = nn.Linear(intermediate_size, output_size)
        self.act_fn = gelu
        # torch.nn.functional.relu
        self.LayerNorm = BERTLayerNorm(config)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.affine(hidden_states)
        return hidden_states


class BERTLayerNorm(nn.Module):
    def __init__(self, config, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BERTLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class DROP_Model(nn.Module):

    def __init__(self, bert_config=None, config=None):
        super(DROP_Model, self).__init__()
        # self.config 指的是 hparams
        self.bert = BertModel(bert_config)
        # 加入adapter
        # adapter_name = self.bert.load_adapter("AdapterHub/bert-base-uncased-pf-drop", source="hf")
        # self.bert.active_adapters = adapter_name
        self.config =config
        self._passage_affine = nn.Linear(self.config.hidden_size, 1)
        self._question_affine = nn.Linear(self.config.hidden_size, 1)
        self.answering_abilities = self.config.answering_abilities,
        self._answer_ability_predictor = BertFeedForward(self.config, 3 * self.config.hidden_size,
                                                         self.config.hidden_size,
                                                         len(self.answering_abilities))

        # 对应论文中 3.2 Multi-Type Answer Predictor
        self.number_pos = -1
        self.base_pos = -2
        self.end_pos = -3
        self.start_pos = -4

        if "span_extraction" in self.answering_abilities[0]:
            self._span_extraction_index = self.answering_abilities[0].index("span_extraction")
            self._base_predictor = BertFeedForward(self.config, self.config.hidden_size, self.config.hidden_size, 1)
            self._start_predictor = BertFeedForward(self.config, self.config.hidden_size, self.config.hidden_size, 1)
            self._end_predictor = BertFeedForward(self.config, self.config.hidden_size, self.config.hidden_size, 1)
            self._start_affine = nn.Linear(4 * self.config.hidden_size, 1)
            self._end_affine = nn.Linear(4 * self.config.hidden_size, 1)
            self._span_number_predictor = BertFeedForward(self.config, 3 * self.config.hidden_size,
                                                          self.config.hidden_size,
                                                          self.config.max_number_of_answer)
        # 加减法
        if "addition_subtraction" in self.answering_abilities[0]:
            self._addition_subtraction_index = self.answering_abilities[0].index("addition_subtraction")
            self._number_sign_predictor = BertFeedForward(self.config, 5 * self.config.hidden_size,
                                                          self.config.hidden_size, 3)
            self._sign_embeddings = nn.Embedding(3, 2 * self.config.hidden_size)
            self._sign_rerank_affine = nn.Linear(2 * self.config.hidden_size, 1)
            self._sign_rerank_predictor = BertFeedForward(self.config, 5 * self.config.hidden_size,
                                                          self.config.hidden_size, 1)

        if "counting" in self.answering_abilities[0]:
            self._counting_index = self.answering_abilities[0].index("counting")
            self._number_count_affine = nn.Linear(2 * self.config.hidden_size, 1)
            self._number_count_predictor = BertFeedForward(self.config, 5 * self.config.hidden_size,
                                                           self.config.hidden_size, 10)

        if "negation" in self.answering_abilities[0]:
            self._negation_index = self.answering_abilities[0].index("negation")
            self._number_negation_predictor = BertFeedForward(self.config, 5 * self.config.hidden_size,
                                                              self.config.hidden_size, 2)

        # 初始化部分 暂时不知道如何使用
        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=self.config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(init_weights)

    def forward(self, mode, input_ids, token_type_ids, attention_mask, number_indices,
                answer_as_span_starts=None, answer_as_span_ends=None, answer_as_span_numbers=None,
                answer_as_counts=None, answer_as_add_sub_expressions=None, answer_as_negations=None,
                number_indices2=None, sign_indices=None, sign_labels=None, encoded_numbers_input=None,
                passage_input=None, question_input=None, pooled_input=None):
        if mode == "rerank_inference":
            assert number_indices2 is not None and sign_indices is not None and encoded_numbers_input is not None and \
                   passage_input is not None and question_input is not None and pooled_input is not None
            sign_encoded_numbers = encoded_numbers_input.unsqueeze(1).repeat(1, number_indices2.size(1), 1, 1)

            sign_mask = (number_indices2 != -1).long()
            clamped_number_indices2 = replace_masked_values(number_indices2, sign_mask, 0)
            sign_output = torch.gather(sign_encoded_numbers, 2,
                                       clamped_number_indices2.unsqueeze(-1).expand(-1, -1, -1,
                                                                                    sign_encoded_numbers.size(-1)))

            clamped_sign_indices = replace_masked_values(sign_indices, sign_mask, 0)
            sign_embeddings = self._sign_embeddings(clamped_sign_indices)
            sign_output += sign_embeddings

            sign_weights = self._sign_rerank_affine(sign_output).squeeze(-1)
            sign_pooled_output = get_self_att_representation(sign_output, sign_weights, sign_mask, dim=2)

            sign_pooled_output = torch.cat(
                [sign_pooled_output, passage_input.unsqueeze(1).repeat(1, sign_pooled_output.size(1), 1),
                 question_input.unsqueeze(1).repeat(1, sign_pooled_output.size(1), 1),
                 pooled_input.unsqueeze(1).repeat(1, sign_pooled_output.size(1), 1)], -1)

            sign_rerank_logits = self._sign_rerank_predictor(sign_pooled_output).squeeze(-1)
            return sign_rerank_logits

        elif mode == "normal":
            span_start_log_probs = None
            span_end_log_probs = None
            span_number_log_probs = None
            number_sign_log_probs = None
            number_mask = None
            sign_rerank_logits = None
            number_negation_log_probs = None
            count_number_log_probs = None
            answer_ability_log_probs = None
            span_start_logits = None
            span_end_logits = None
            encoded_numbers = None
            number_sign_logits = None
            best_span_number = None

            all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)

            passage_weights = self._passage_affine(all_encoder_layers[self.base_pos]).squeeze(-1)
            passage_vector = get_self_att_representation(all_encoder_layers[self.base_pos], passage_weights,
                                                         token_type_ids)

            question_weights = self._question_affine(all_encoder_layers[self.base_pos]).squeeze(-1)
            question_vector = get_self_att_representation(all_encoder_layers[self.base_pos], question_weights,
                                                          (1 - token_type_ids))

            best_answer_ability, best_count_number, best_negations_for_numbers = None, None, None
            if len(self.answering_abilities[0]) >= 1:
                # Shape: (batch_size, number_of_abilities)
                answer_ability_logits = \
                    self._answer_ability_predictor(torch.cat([passage_vector, question_vector, pooled_output], -1))
                answer_ability_log_probs = log_softmax(answer_ability_logits, -1)
                best_answer_ability = torch.argmax(answer_ability_log_probs, -1)

                # Shape: (batch_size, # of numbers in the passage)
                number_indices = number_indices.squeeze(-1)
                number_mask = (number_indices != -1).long()

                if "counting" in self.answering_abilities[0]:
                    # Shape: (batch_size, # of numbers in the passage, 2*hidden_size)
                    encoded_passage_for_numbers = torch.cat(
                        [all_encoder_layers[self.base_pos], all_encoder_layers[self.number_pos]], dim=-1)
                    encoded_numbers = gather_representations(encoded_passage_for_numbers, number_indices)

                    # Shape: (batch_size, hidden_size)
                    count_weights = self._number_count_affine(encoded_numbers).squeeze(-1)
                    count_pooled_output = get_self_att_representation(encoded_numbers, count_weights, number_mask)

                    # Shape: (batch_size, 10)
                    count_number_logits = self._number_count_predictor(
                        torch.cat([count_pooled_output, passage_vector, question_vector, pooled_output], -1))
                    count_number_log_probs = torch.nn.functional.log_softmax(count_number_logits, -1)

                    # Info about the best count number prediction
                    # Shape: (batch_size,)
                    best_count_number = torch.argmax(count_number_log_probs, -1)

                if "span_extraction" in self.answering_abilities[0]:
                    base_weights = self._base_predictor(all_encoder_layers[self.base_pos]).squeeze(-1)
                    base_q_pooled_output = get_self_att_representation(all_encoder_layers[self.base_pos],
                                                                       base_weights, (1 - token_type_ids))

                    start_weights = self._start_predictor(all_encoder_layers[self.start_pos]).squeeze(-1)
                    start_q_pooled_output = get_self_att_representation(all_encoder_layers[self.start_pos],
                                                                        start_weights, (1 - token_type_ids))

                    end_weights = self._end_predictor(all_encoder_layers[self.end_pos]).squeeze(-1)
                    end_q_pooled_output = get_self_att_representation(all_encoder_layers[self.end_pos],
                                                                      end_weights, (1 - token_type_ids))

                    start_output = torch.cat((all_encoder_layers[self.base_pos], all_encoder_layers[self.start_pos],
                                              base_q_pooled_output.unsqueeze(1) * all_encoder_layers[self.base_pos],
                                              start_q_pooled_output.unsqueeze(1) * all_encoder_layers[self.start_pos]),
                                             -1)

                    end_output = torch.cat((all_encoder_layers[self.base_pos], all_encoder_layers[self.end_pos],
                                            base_q_pooled_output.unsqueeze(1) * all_encoder_layers[self.base_pos],
                                            end_q_pooled_output.unsqueeze(1) * all_encoder_layers[self.end_pos]), -1)

                    span_start_logits = self._start_affine(start_output).squeeze(-1)
                    span_end_logits = self._end_affine(end_output).squeeze(-1)

                    span_start_log_probs = torch.nn.functional.log_softmax(span_start_logits, -1)
                    span_end_log_probs = torch.nn.functional.log_softmax(span_end_logits, -1)

                    # Shape: (batch_size, 8)
                    span_number_logits = self._span_number_predictor(
                        torch.cat([passage_vector, question_vector, pooled_output], -1))
                    span_number_log_probs = torch.nn.functional.log_softmax(span_number_logits, -1)

                    # Info about the best count number prediction
                    # Shape: (batch_size,)
                    best_span_number = torch.argmax(span_number_log_probs, -1)

                if "addition_subtraction" in self.answering_abilities[0]:
                    # Shape: (batch_size, # of numbers in the passage, 2*hidden_size)
                    encoded_passage_for_numbers = torch.cat(
                        [all_encoder_layers[self.base_pos], all_encoder_layers[self.number_pos]], dim=-1)
                    # encoded_passage_for_numbers = torch.tensor(encoded_passage_for_numbers)
                    encoded_numbers = gather_representations(encoded_passage_for_numbers, number_indices)

                    # Shape: (batch_size, # of numbers in the passage, 5*hidden_size)
                    concat_encoded_numbers = torch.cat(
                        [encoded_numbers, passage_vector.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1),
                         question_vector.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1),
                         pooled_output.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1)], -1)

                    # Shape: (batch_size, # of numbers in the passage, 3)
                    number_sign_logits = self._number_sign_predictor(concat_encoded_numbers)

                    number_sign_log_probs = torch.nn.functional.log_softmax(number_sign_logits, -1)

                    # Rerank
                    if number_indices2 is not None and sign_indices is not None:
                        # Shape: (batch_size, beam_size, # of numbers in the passage, 2*hidden_size)
                        sign_encoded_numbers = encoded_numbers.unsqueeze(1).repeat(1, number_indices2.size(1), 1, 1)

                        # Shape: (batch_size, beam_size, max_count)
                        sign_mask = (number_indices2 != -1).long()
                        clamped_number_indices2 = replace_masked_values(number_indices2, sign_mask, 0)
                        # Shape: (batch_size, beam_size, max_count, 2*hidden_size)
                        sign_output = torch.gather(sign_encoded_numbers, 2,
                                                   clamped_number_indices2.unsqueeze(-1).expand(-1, -1, -1,
                                                                                                sign_encoded_numbers.size(
                                                                                                    -1)))

                        # Shape: (batch_size, beam_size, max_count, 2*hidden_size)
                        clamped_sign_indices = replace_masked_values(sign_indices, sign_mask, 0)
                        sign_embeddings = self._sign_embeddings(clamped_sign_indices)
                        sign_output += sign_embeddings

                        # Shape: (batch_size, beam_size, 2*hidden_size)
                        sign_weights = self._sign_rerank_affine(sign_output).squeeze(-1)
                        sign_pooled_output = get_self_att_representation(sign_output, sign_weights, sign_mask, dim=2)

                        sign_pooled_output = torch.cat(
                            [sign_pooled_output, passage_vector.unsqueeze(1).repeat(1, sign_pooled_output.size(1), 1),
                             question_vector.unsqueeze(1).repeat(1, sign_pooled_output.size(1), 1),
                             pooled_output.unsqueeze(1).repeat(1, sign_pooled_output.size(1), 1)], -1)

                        # Shape: (batch_size, beam_size)
                        sign_rerank_logits = self._sign_rerank_predictor(sign_pooled_output).squeeze(-1)

                if "negation" in self.answering_abilities[0]:
                    # Shape: (batch_size, # of numbers in the passage, 2*hidden_size)
                    encoded_passage_for_numbers = torch.cat(
                        [all_encoder_layers[self.base_pos], all_encoder_layers[self.number_pos]], dim=-1)
                    encoded_numbers = gather_representations(encoded_passage_for_numbers, number_indices)

                    # Shape: (batch_size, # of numbers in the passage, 5*hidden_size)
                    concat_encoded_numbers = torch.cat(
                        [encoded_numbers, passage_vector.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1),
                         question_vector.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1),
                         pooled_output.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1)], -1)

                    # Shape: (batch_size, # of numbers in the passage, 2)
                    number_negation_logits = self._number_negation_predictor(concat_encoded_numbers)

                    number_negation_log_probs = functional.log_softmax(number_negation_logits, -1)

                    # Shape: (batch_size, # of numbers in passage).
                    best_negations_for_numbers = torch.argmax(number_negation_log_probs, -1)
                    # For padding numbers, the best sign masked as 0 (not included).
                    best_negations_for_numbers = replace_masked_values(best_negations_for_numbers, number_mask, 0)

            # If answer is given, compute the loss.
            if (
                    answer_as_span_starts is not None and answer_as_span_ends is not None and answer_as_span_numbers is not None) \
                    or answer_as_counts is not None or answer_as_add_sub_expressions is not None or answer_as_negations is not None \
                    or (number_indices2 is not None and sign_indices is not None and sign_labels is not None):

                log_marginal_likelihood_list = []

                for answering_ability in self.answering_abilities[0]:
                    if answering_ability == "span_extraction":
                        # Shape: (batch_size, # of answer spans)
                        log_likelihood_span_starts = gather_log_likelihood(span_start_log_probs, answer_as_span_starts)
                        log_likelihood_span_ends = gather_log_likelihood(span_end_log_probs, answer_as_span_ends)
                        log_likelihood_span_starts = masked_logsumexp(log_likelihood_span_starts, answer_as_span_starts)
                        log_likelihood_span_ends = masked_logsumexp(log_likelihood_span_ends, answer_as_span_ends)

                        log_likelihood_span_numbers = gather_log_likelihood(span_number_log_probs,
                                                                            answer_as_span_numbers)
                        log_likelihood_span_numbers = masked_logsumexp(log_likelihood_span_numbers,
                                                                       answer_as_span_numbers)

                        log_marginal_likelihood_for_span = log_likelihood_span_starts + log_likelihood_span_ends + \
                                                           log_likelihood_span_numbers
                        # Shape: (batch_size, )
                        log_marginal_likelihood_list.append(log_marginal_likelihood_for_span)

                    elif answering_ability == "addition_subtraction":
                        # The padded add-sub combinations use -1 as the signs for all numbers, and we mask them here.
                        # Shape: (batch_size, # of combinations, # of numbers in the passage)
                        gold_add_sub_mask = (answer_as_add_sub_expressions != -1).long()
                        clamped_gold_add_sub_signs = replace_masked_values(answer_as_add_sub_expressions,
                                                                           gold_add_sub_mask, 0)
                        # Shape: (batch_size, # of numbers in the passage, # of combinations)
                        gold_add_sub_signs = clamped_gold_add_sub_signs.transpose(1, 2)
                        # Shape: (batch_size, # of numbers in the passage, # of combinations)
                        log_likelihood_for_number_signs = torch.gather(number_sign_log_probs, 2, gold_add_sub_signs)
                        # the log likelihood of the masked positions should be 0
                        # so that it will not affect the joint probability
                        log_likelihood_for_number_signs = \
                            replace_masked_values(log_likelihood_for_number_signs, number_mask.unsqueeze(-1), 0)
                        # Shape: (batch_size, # of combinations)
                        log_likelihood_for_add_subs = log_likelihood_for_number_signs.sum(1)
                        # For those padded combinations, we set their log probabilities to be very small negative value
                        # Shape: (batch_size, # of combinations)
                        gold_combination_mask = (gold_add_sub_mask.sum(-1) != 0).long()
                        log_likelihood_for_add_subs = \
                            replace_masked_values(log_likelihood_for_add_subs, gold_combination_mask, -1e7)
                        # Shape: (batch_size, )
                        log_likelihood_for_add_sub = logsumexp(log_likelihood_for_add_subs)
                        # Shape: (batch_size, )
                        log_likelihood_sign_rerank = distant_cross_entropy(sign_rerank_logits, sign_labels)

                        log_marginal_likelihood_for_add_sub = log_likelihood_for_add_sub + log_likelihood_sign_rerank
                        log_marginal_likelihood_list.append(log_marginal_likelihood_for_add_sub)

                    elif answering_ability == "counting":
                        # Shape: (batch_size, # of count answers)
                        log_likelihood_for_counts = gather_log_likelihood(count_number_log_probs, answer_as_counts)
                        # Shape: (batch_size, )
                        log_marginal_likelihood_for_count = masked_logsumexp(log_likelihood_for_counts,
                                                                             answer_as_counts)
                        log_marginal_likelihood_list.append(log_marginal_likelihood_for_count)

                    elif answering_ability == "negation":
                        # The padded add-sub combinations use -1 as the signs for all numbers, and we mask them here.
                        # Shape: (batch_size, # of combinations, # of numbers in the passage)
                        gold_negation_mask = (answer_as_negations != -1).long()
                        clamped_gold_negations = replace_masked_values(answer_as_negations, gold_negation_mask, 0)
                        # Shape: (batch_size, # of numbers in the passage, # of combinations)
                        gold_negations = clamped_gold_negations.transpose(1, 2)
                        # Shape: (batch_size, # of numbers in the passage, # of combinations)
                        log_likelihood_for_negations = torch.gather(number_negation_log_probs, 2, gold_negations)
                        # the log likelihood of the masked positions should be 0
                        # so that it will not affect the joint probability
                        log_likelihood_for_negations = \
                            replace_masked_values(log_likelihood_for_negations, number_mask.unsqueeze(-1), 0)
                        # Shape: (batch_size, # of combinations)
                        log_likelihood_for_negations = log_likelihood_for_negations.sum(1)
                        # For those padded combinations, we set their log probabilities to be very small negative value
                        # Shape: (batch_size, # of combinations)
                        gold_combination_mask = (gold_negation_mask.sum(-1) != 0).long()
                        log_likelihood_for_negations = \
                            replace_masked_values(log_likelihood_for_negations, gold_combination_mask, -1e7)
                        # Shape: (batch_size, )
                        log_marginal_likelihood_for_negation = logsumexp(log_likelihood_for_negations)
                        log_marginal_likelihood_list.append(log_marginal_likelihood_for_negation)

                    else:
                        raise ValueError(f"Unsupported answering ability: {answering_ability}")

                if len(self.answering_abilities[0]) > 1:
                    # Add the ability probabilities if there are more than one abilities
                    all_log_marginal_likelihoods = torch.stack(log_marginal_likelihood_list, dim=-1)
                    all_log_marginal_likelihoods = all_log_marginal_likelihoods + answer_ability_log_probs
                    marginal_log_likelihood = logsumexp(all_log_marginal_likelihoods)
                else:
                    marginal_log_likelihood = log_marginal_likelihood_list[0]
                return - marginal_log_likelihood.mean()

            else:
                output_dict = {}
                if len(self.answering_abilities[0]) >= 1:
                    output_dict["best_answer_ability"] = best_answer_ability
                if "counting" in self.answering_abilities[0]:
                    output_dict["best_count_number"] = best_count_number
                if "negation" in self.answering_abilities[0]:
                    output_dict["best_negations_for_numbers"] = best_negations_for_numbers
                if "span_extraction" in self.answering_abilities[0]:
                    output_dict["span_start_logits"] = span_start_logits
                    output_dict["span_end_logits"] = span_end_logits
                    output_dict["best_span_number"] = best_span_number
                if "addition_subtraction" in self.answering_abilities[0]:
                    output_dict["number_sign_logits"] = number_sign_logits
                    output_dict["number_mask"] = number_mask
                    output_dict["encoded_numbers_output"] = encoded_numbers
                    output_dict["passage_output"] = passage_vector
                    output_dict["question_output"] = question_vector
                    output_dict["pooled_output"] = pooled_output
                return output_dict


        else:
            raise Exception
