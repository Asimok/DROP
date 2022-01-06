import logging
import sys

from tqdm import tqdm

from config.hparams import PARAMS
from dataset_dataloaders.drop_dataloader import DropDataloader

sys.path.append('/data2/maqi/drop')
import os

# 指定对程序可见的GPU编号
# 若使用服务器多卡训练
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import collections

import torch
from torch.utils.data import DataLoader



hparams = PARAMS
hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)


train_dataset = DropDataloader(hparams, evaluate=False)

# train_dataloader = DataLoader(
#     train_dataset,
#     batch_size=batch_size,
#     num_workers=hparams.workers,
#     shuffle=True,
#     drop_last=True
# )
#
# tqdm_batch_iterator = tqdm(train_dataloader)
# data = []
# i = 0
# for batch_idx, batch in enumerate(tqdm_batch_iterator):
#     print(batch_idx)
#     data.append(batch)
#     # break
#     # print(batch.keys())
#     i += 1
#     if i > 5:
#         break
