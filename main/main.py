# 指定对程序可见的GPU编号
# 若使用服务器多卡训练
import collections
import os
import sys

from config.hparams import PARAMS
from model.trainer import Trainer
from tools.log import get_logger

sys.path.append('/data0/maqi/drop')
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

hparams = PARAMS
hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)

n_gpu = 1
device = 'cuda'
global_step = 1
best_f1 = 0
epoch = 10

trainer = Trainer(hparams=hparams, mode='train')
log = get_logger(log_name="main")
log.info("***** Prepare for Train *****")
log.info("Num Epochs = %d", int(hparams.train_epochs))
log.info("batch size = %d", int(hparams.batch_size_for_train))
for i in range(hparams.train_epochs):
    global_step, model, best_f1 = trainer.run_train_epoch(n_gpu, device, global_step=global_step, best_f1=best_f1,
                                                          epoch=i + 1)
    print(global_step)
    if i % 10 == 0:
        trainer.run_predict()
