from transformers import AutoModelWithHeads, AutoConfig, BertModel

# model = AutoModelWithHeads.from_pretrained("bert-base-uncased")
# adapter_name = model.load_adapter("AdapterHub/bert-base-uncased-pf-drop", source="hf")
# model.active_adapters = adapter_name

# model = AutoModelWithHeads.from_pretrained("roberta-base")
# adapter_name = model.load_adapter("AdapterHub/roberta-base-pf-drop", source="hf")
# model.active_adapters = adapter_name
#

bert = AutoModelWithHeads.from_pretrained("/data0/maqi/pretrained_model/pytorch/bert-base-uncased")
adapter_name = bert.load_adapter("AdapterHub/bert-base-uncased-pf-drop", source="hf")
bert.active_adapters = adapter_name

config = AutoConfig.from_pretrained("/data0/maqi/pretrained_model/pytorch/bert-base-uncased")

bert1 = BertModel(config, add_pooling_layer=False)
