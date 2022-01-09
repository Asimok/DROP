from transformers import BertPreTrainedModel, BertModel


class DROP_Model(BertPreTrainedModel):

    def __init__(self, config=None, hparams=None):
        super().__init__(config)
        self.hparams = hparams
        self.config = config

        self.bert = BertModel(self.config, add_pooling_layer=False)
        adapter_name = self.bert.load_adapter("AdapterHub/bert-base-uncased-pf-drop", source="hf")
        self.bert.active_adapters = adapter_name

        
