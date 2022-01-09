from torch.utils.data import Dataset

from dataset_dataloaders.process_source_data_for_drop import load_dataset


class DropDataloader(Dataset):

    def __init__(self, hparams=None, evaluate: bool = False, tokenizer=None) -> None:
        self.hparams = hparams
        self.evaluate = evaluate
        self.tokenizer = tokenizer
        self.dataset, self.examples = load_dataset(_hparams=hparams, _tokenizer=self.tokenizer, _evaluate=self.evaluate)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def get_examples(self):
        return self.examples
