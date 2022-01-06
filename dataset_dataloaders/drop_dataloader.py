from torch.utils.data import Dataset

from dataset_dataloaders.process_source_data_for_drop import load_dataset
from tools.log import get_logger




class DropDataloader(Dataset):

    def __init__(self, hparams=None, evaluate: bool = False, tokenizer=None) -> None:
        self.hparams = hparams
        self.evaluate = evaluate
        self.tokenizer = tokenizer
        self.log = get_logger(log_name="DropDataloader")
        self.dataset, self.examples = load_dataset(_hparams=hparams, _tokenizer=self.tokenizer, _evaluate=self.evaluate)

    def __getitem__(self, index):
        return self.dataset[index]

    def get_examples(self):
        return self.examples
