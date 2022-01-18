import os
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# 超参数
PARAMS = defaultdict(
    # model
    do_train=True,
    do_test=True,
    max_count=10000,  # maximum counting ability of the network
    answering_abilities=["span_extraction", "addition_subtraction", "counting", "negation"],

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
    pretrainedModelPath='/data0/maqi/pretrained_model/pytorch/bert-base-uncased',
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
    max_question_add_passage_length=512,  # 问题+段落长度
    max_number_indices_len=64,  # 包含的数字数量
    max_answer_nums_length=4,  # 答案数量
    max_auxiliary_len=1,  # 辅助计算的字段长度 比如 id count
    max_add_sub_combination_len=8,  # 加减运算的数量
    max_negation_combination_len=1,  # 否定句数量
    max_number_of_numbers_to_consider=2,  # 算术运算数字的最大数量

)
