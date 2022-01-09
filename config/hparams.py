import os
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# 超参数
PARAMS = defaultdict(
    # model
    do_train=True,
    do_test=True,
    max_count=10000,  # maximum counting ability of the network

    # train
    device='cuda',
    gpu_ids=[0, 1],  # 注意：在程序中设置gpu_id仍要从0开始，gpu_ids为 CUDA_VISIBLE_DEVICES 的索引
    workers=0,
    batch_size_for_train=8,
    batch_size_for_test=16,
    train_epochs=10,
    ## optimizer
    learning_rate=3e-5,
    warmup_proportion=0.05,

    # 206
    ## raw files
    EnvironmentPath='/data0/maqi/drop',
    datasetPath='/data0/maqi/drop/datasets/simplified/drop_data',
    # datasetPath='/data0/maqi/drop/datasets/drop_data',
    trainFile='drop_dataset_train_standardized.json',
    testFile='drop_dataset_dev_standardized.json',
    pretrainedModelPath='bert-base-uncased',
    ## processed files
    modelSavePath='/data0/maqi/drop/output/model',
    cachePath='/data0/maqi/drop/temp',  # 预处理数据的缓存
    logPath='/data0/maqi/drop/output/logs',
    logFile='log.txt',
    output_dir='test_model',  # 不同模型的日志保存目录
    tensorboard_path='runs',  # logPath + output_dir + tensorboard_path + date
    train_eval_file='/data0/maqi/drop/datasets/processed/drop_data/train_eval.json',
    dev_eval_file='/data0/maqi/drop/datasets/processed/drop_data/dev_eval.json',

    ## tokenize
    # 如果更改下面参数 则需要重新进行数据集预处理
    max_passage_length=512,  # 段落长度
    max_question_length=128,  # 问题长度
    max_answer_span_length=32,  # 答案长度 包括multi span
    max_auxiliary_len=8,  # 辅助计算的字段长度 比如 id count

)
