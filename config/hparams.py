from collections import defaultdict

# 超参数
PARAMS = defaultdict(
    # 全局参数
    do_train=True,
    do_test=True,

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
    cachePath='/data0/maqi/drop/temp',
    logPath='/data0/maqi/drop/output/logs',
    logFile='log.txt',
    train_eval_file='/data0/maqi/drop/datasets/processed/drop_data/train_eval.json',
    dev_eval_file='/data0/maqi/drop/datasets/processed/drop_data/dev_eval.json',

    # model
    max_count=10000,  # maximum counting ability of the network

    ## tokenize
    max_passage_length=512,
    max_question_length=128,
    max_answer_span_length=32,
    max_auxiliary_len=8,

)
