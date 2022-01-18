import numpy as np
import torch
from transformers import CTRLTokenizer, CTRLLMHeadModel

tokenizer = CTRLTokenizer.from_pretrained('ctrl')
model = CTRLLMHeadModel.from_pretrained('ctrl').cuda()

inputs_text="I have a dog, I haste me to my bed,"
inputs = torch.tensor([tokenizer(inputs_text)["input_ids"]],dtype=torch.long).cuda()



