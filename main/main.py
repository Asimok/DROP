# 指定对程序可见的GPU编号
# 若使用服务器多卡训练
import collections
import os

from config.hparams import PARAMS
from model.trainer import Trainer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

hparams = PARAMS
hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)

n_gpu = 1
device = 'cuda'
global_step = 1
best_f1 = 0
epoch = 10
trainer = Trainer(hparams=hparams, mode='main')
global_step, model, best_f1 = trainer.run_train_epoch(n_gpu, device, global_step, best_f1, epoch)
