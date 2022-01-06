import json
import random

read_train_file = '../datasets/drop_data/drop_dataset_train_standardized.json'
read_dev_file = '../datasets/drop_data/drop_dataset_dev_standardized.json'
out_train_file = '../datasets/simplified/drop_data/drop_dataset_train_standardized.json'
out_dev_train = '../datasets/simplified/drop_data/drop_dataset_dev_standardized.json'

train_data = json.load(open(read_train_file, 'r'))
dev_data = json.load(open(read_dev_file, 'r'))


def split_dataset(original_data, max_len=10):
    # 分割
    simplified_data = {}
    data_keys = list(original_data.keys())
    # 随机下标
    for i in range(max_len):
        loc = random.randint(0, len(data_keys))
        while simplified_data.get(data_keys[loc]):
            loc = random.randint(0, len(data_keys))
        simplified_data[data_keys[loc]] = original_data[data_keys[loc]]
    return simplified_data

def save_file(outpath,data):
    with open(outpath, 'w') as f:
        json.dump(data, f)


simplified_train_data = split_dataset(train_data, 20)
simplified_dev_data = split_dataset(dev_data, 10)

save_file(out_train_file,simplified_train_data)
save_file(out_dev_train,simplified_dev_data)