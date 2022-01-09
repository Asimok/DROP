import json

vocab_data = json.load(open('/data0/maqi/pretrained_model/pytorch/roberta-base/vocab.json','r'))
output_txt = '/data0/maqi/pretrained_model/pytorch/roberta-base/vocab.txt'
with open('/data0/maqi/pretrained_model/pytorch/bert-base-uncased/vocab.txt','r') as f:
    d =f.readlines()
with open(output_txt,'r') as f:
    m =f.readlines()

vocab_data_list = list(vocab_data.keys())
with open(output_txt,'w') as f:
    for i in vocab_data_list:
        f.writelines(i+'\n')