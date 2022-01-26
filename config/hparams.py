import os
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 超参数
PARAMS = defaultdict(
    # main
    do_train=True,
    do_test=True,

    # train
    device='cuda',
    gpu_ids=[0],  # 注意：在程序中设置gpu_id仍要从0开始，gpu_ids为 CUDA_VISIBLE_DEVICES 的索引
    workers=0,
    batch_size_for_train=4,
    batch_size_for_test=8,
    train_epochs=100,

    ## optimizer
    seed=42,
    learning_rate=3e-5,
    warmup_proportion=0.05,
    # model
    # 768
    hidden_size=1024,
    initializer_range=0.02,
    # BERT里的参数 The sttdev of the truncated_normal_initializer for initializing all weight matrices.
    max_count=4,  # maximum counting ability of the network
    answering_abilities=["span_extraction", "addition_subtraction", "counting", "negation"],

    beam_size=3,
    gradient_accumulation_steps=2,
    length_heuristic=0.05,  # Weight on length heuristic.
    n_best_size=20,  # The total number of n-best predictions to generate in the nbest_predictions.json output file.
    max_answer_length=30,
    # The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.
    do_lower_case=False,
    # Whether to lower case the input text. Should be True for uncased models and False for cased models.
    verbose_logging=False,
    # If true, all of the warnings related to data processing will be printed. A number of warnings are expected for a normal SQuAD evaluation.

    # 206
    ## raw files
    # EnvironmentPath='/data0/maqi/drop',
    # datasetPath='/data0/maqi/drop/datasets/simplified/drop_data',
    datasetPath='/data0/maqi/drop/datasets/drop_data',
    trainFile='drop_dataset_train_standardized.json',
    testFile='drop_dataset_dev_standardized.json',
    # pretrainedModelPath='/data0/maqi/pretrained_model/pytorch/bert-base-uncased',
    pretrainedModelPath='/data0/maqi/pretrained_model/pytorch/bert-large-uncased',


    ## processed files
    modelSavePath='/data0/maqi/drop/output/model',
    cachePath='/data0/maqi/drop/temp',  # 预处理数据的缓存

    output_dir='/data0/maqi/drop/output',  # 不同模型的日志保存目录
    current_model='test_model_large',
    logPath='logs/log.txt',  # output_dir + current_model + logPath
    tensorboard_path='tensorboard_runs',  # output_dir + current_model + tensorboard_path + date
    best_model_save_path='saved_model/checkpoint.pth.tar',  # output_dir + current_model + best_model_save_path
    prediction_file_path='predictions.json',  # output_dir + current_model
    performance_file_path='performance.txt',  # output_dir + current_model
    model_config_file_path='config.txt',  # output_dir + current_model

    # train_eval_file='/data0/maqi/drop/datasets/processed/drop_data/train_eval.json',
    # dev_eval_file='/data0/maqi/drop/datasets/processed/drop_data/dev_eval.json',

    ## tokenize
    # 如果更改下面参数 则需要重新进行数据集预处理
    max_question_add_passage_length=512,  # 问题+段落长度
    max_number_indices_len=256,  # 包含的数字数量
    max_answer_nums_length=2,  # 答案数量
    max_auxiliary_len=1,  # 辅助计算的字段长度 比如 id count
    max_add_sub_combination_len=4,  # 加减运算的数量
    max_negation_combination_len=1,  # 否定句数量
    max_number_of_numbers_to_consider=2,  # 算术运算数字的最大数量
    max_number_of_answer=8,

)
