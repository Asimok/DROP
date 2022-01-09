import collections

from config.hparams import PARAMS
from model.trainer import Trainer

hparams = PARAMS
hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)
Trainer = Trainer(hparams=hparams, mode='train')
