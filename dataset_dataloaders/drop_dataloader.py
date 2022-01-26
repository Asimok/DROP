from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from dataset_dataloaders.process_source_data_for_drop import load_dataset

def my_collate(batch):
    tensors_list, features_list = [], []
    for dataset_data in batch:
        tensors_list.append(dataset_data[0])
        features_list.append(dataset_data[1])
    tensors_list = default_collate(tensors_list)

    return tensors_list, features_list

class DropDataloader(Dataset):

    def __init__(self, hparams=None, evaluate: bool = False, tokenizer=None, file_path=None) -> None:
        self.hparams = hparams
        self.evaluate = evaluate
        self.tokenizer = tokenizer
        # self.datasets, self.dataset_features,self.examples = load_dataset(_hparams=hparams, _tokenizer=self.tokenizer, _evaluate=self.evaluate,file_path=file_path)
        self.datasets ,self.examples= load_dataset(_hparams=hparams,
                                     _tokenizer=self.tokenizer,
                                     _evaluate=self.evaluate,
                                     file_path=file_path)

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

    def get_examples(self):
        return self.examples

