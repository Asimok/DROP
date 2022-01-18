import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


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
        super(DROP_Model,self).__init__()
        # config 指的是 hparams
        self.bert = BertModel(bert_config, add_pooling_layer=False)
        # 加入adapter
        adapter_name = self.bert.load_adapter("AdapterHub/bert-base-uncased-pf-drop", source="hf")
        self.bert.active_adapters = adapter_name

        self._passage_affine = nn.Linear(config.hidden_size, 1)
        self._question_affine = nn.Linear(config.hidden_size, 1)
        self.answering_abilities = config.hidden_size,
        self._answer_ability_predictor = BertFeedForward(config, 3 * config.hidden_size, config.hidden_size,
                                                         len(self.answering_abilities))

        # 对应论文中 3.2 Multi-Type Answer Predictor
        self.number_pos = -1
        self.base_pos = -2
        self.end_pos = -3
        self.start_pos = -4

        if "span_extraction" in self.answering_abilities:
            self._span_extraction_index = self.answering_abilities.index("span_extraction")
            self._base_predictor = BertFeedForward(config, config.hidden_size, config.hidden_size, 1)
            self._start_predictor = BertFeedForward(config, config.hidden_size, config.hidden_size, 1)
            self._end_predictor = BertFeedForward(config, config.hidden_size, config.hidden_size, 1)
            self._start_affine = nn.Linear(4 * config.hidden_size, 1)
            self._end_affine = nn.Linear(4 * config.hidden_size, 1)
            self._span_number_predictor = BertFeedForward(config, 3 * config.hidden_size, config.hidden_size,
                                                          config.max_number_of_answer)
        # 加减法
        if "addition_subtraction" in self.answering_abilities:
            self._addition_subtraction_index = self.answering_abilities.index("addition_subtraction")
            self._number_sign_predictor = BertFeedForward(config, 5 * config.hidden_size, config.hidden_size, 3)
            self._sign_embeddings = nn.Embedding(3, 2 * config.hidden_size)
            self._sign_rerank_affine = nn.Linear(2 * config.hidden_size, 1)
            self._sign_rerank_predictor = BertFeedForward(config, 5 * config.hidden_size, config.hidden_size, 1)

        if "counting" in self.answering_abilities:
            self._counting_index = self.answering_abilities.index("counting")
            self._number_count_affine = nn.Linear(2 * config.hidden_size, 1)
            self._number_count_predictor = BertFeedForward(config, 5 * config.hidden_size, config.hidden_size, 10)

        if "negation" in self.answering_abilities:
            self._negation_index = self.answering_abilities.index("negation")
            self._number_negation_predictor = BertFeedForward(config, 5 * config.hidden_size, config.hidden_size, 2)

        # 初始化部分 暂时不知道如何使用
        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        self.apply(init_weights)
