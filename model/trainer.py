import os
from datetime import datetime

import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup

from config.hparams import PARAMS
from dataset_dataloaders.drop_dataloader import DropDataloader
from model.drop_model import DROP_Model
from model.optimization import BERTAdam
from tools.log import get_logger


class Trainer(object):
    def __init__(self, hparams, mode='train'):
        self.log = get_logger(log_name="Trainer")
        self.hparams = hparams
        self.mode = mode
        self.device = self.hparams.device
        self.model = None

        # self.criterion = None
        # self.optimizer = None
        # self.scheduler = None

        self.summery_writer = self.init_SummaryWriter()
        self.pretrained_model_config, self.tokenizer = self.Load_pretrained_model_config()
        # TODO 加载数据根据 mode='train' 参数 减少内存占用
        self.train_dataloader, self.t_total = self.build_dataloader_for_train()
        self.test_dataloader, self.test_examples = self.build_dataloader_for_test()
        self.build_model()
        self.save_train_config()

    def Load_pretrained_model_config(self):
        """
        加载预训练模型参数 并加入针对该模型的自定义参数
        :return: pretrained_model_config, tokenizer
        """
        self.log.info("Load pretrained model from file %s ...", self.hparams.pretrainedModelPath)
        pretrained_model_config = AutoConfig.from_pretrained(self.hparams.pretrainedModelPath)
        tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrainedModelPath)

        # TODO 此处初始化几个自定义参数 用作bert初始化
        # self.pretrained_model_config.context_question_length = self.hparams.max_passage_length + self.hparams.max_question_length
        # self.pretrained_model_config.max_answer_length = self.hparams.max_answer_length
        self.log.info("Load pretrained model config finished!!!")
        return pretrained_model_config, tokenizer

    def build_dataloader_for_train(self):
        self.log.info("Load train dataset from file %s ...",
                      os.path.join(self.hparams.datasetPath, self.hparams.trainFile))

        train_dataset = DropDataloader(hparams=self.hparams, evaluate=False, tokenizer=self.tokenizer)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size_for_train,
            num_workers=self.hparams.workers,
            shuffle=True,
            drop_last=True
        )
        t_total = len(train_dataloader) // self.hparams.batch_size_for_train * self.hparams.train_epochs
        self.log.info("Num steps = %d", t_total)
        self.log.info("Load train dataset finished!!!")
        return train_dataloader, t_total

    def build_dataloader_for_test(self):
        self.log.info("Load test dataset from file %s ...",
                      os.path.join(self.hparams.datasetPath, self.hparams.testFile))
        test_dataset = DropDataloader(hparams=self.hparams, evaluate=True, tokenizer=self.tokenizer)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.hparams.batch_size_for_test,
            num_workers=self.hparams.workers,
            shuffle=True,
            drop_last=True
        )
        test_examples = test_dataset.get_examples()
        self.log.info("Load train dataset finished!!!")
        return test_dataloader, test_examples

    def build_model(self):
        # Define model
        self.log.info("Define model...")
        self.model = DROP_Model.from_pretrained(self.hparams.pretrainedModelPath, from_tf=False,
                                                config=self.pretrained_model_config,
                                                hparams=self.hparams)
        # 随即初始化
        self.model.init_weights()

        # GPU or CPU
        if not torch.cuda.is_available():
            self.device = 'cpu'
        self.log.info('use =%s to train', self.device)
        self.model.to(self.device)

        # Use Multi-GPUs
        if len(self.hparams.gpu_ids) > 1 and self.device != 'cpu':
            self.model = nn.DataParallel(self.model, self.hparams.gpu_ids)
            self.log.info("Use Multi-GPUs" + str(self.hparams.gpu_ids))
        else:
            self.log.info("Use one GPU")

        # Define Loss and Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        param_optimizer = list(self.model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}
        ]
        optimizer = BERTAdam(optimizer_grouped_parameters,
                             lr=self.hparams.learning_rate,
                             warmup=self.hparams.warmup_proportion,
                             t_total=self.t_total)
        criterion = nn.CrossEntropyLoss()
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=self.hparams.warmup_proportion, num_training_steps=self.t_total
        # )

        self.log.info("Define model finished!!!")
        return criterion, optimizer

    def save_train_config(self):
        """
        保存训练模型时的参数值
        :return:
        """
        # performance_path = os.path.join(log_path, 'performance.txt')
        log_path = os.path.join(self.hparams.logPath, self.hparams.output_dir)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.log.info('output_dir: {}'.format(log_path))
        config_path = os.path.join(log_path, 'config.txt')
        # 保存config
        with open(config_path, 'w') as f:
            config_fields = PARAMS
            for k, v in config_fields:
                f.write("{}: {}".format(k, v))

    def save_best_model(self, model, optimizer, global_step, epoch):
        """
        保存训练过程中 性能最佳的模型
        :param model:
        :param optimizer:
        :param global_step:
        :param epoch:
        :return:
        """
        self.log.info("prepared to save best model")
        log_path = os.path.join(self.hparams.logPath, self.hparams.output_dir)
        save_path = os.path.join(log_path, 'checkpoint.pth.tar')  # 最优模型保存路径模型
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': global_step,
            'epoch': epoch
        }, save_path)
        self.log.info(" best model have saved to " + save_path)

    def init_SummaryWriter(self):
        """
        初始化 SummaryWriter
        :return: SummaryWriter
        """
        today = str(datetime.today().month) + 'm' + str(
            datetime.today().day) + 'd_' + str(datetime.today().hour) + 'h-' + str(datetime.today().minute) + 'm'
        log_path = os.path.join(self.hparams.logPath, self.hparams.output_dir)
        writer_path = os.path.join(log_path, self.hparams.tensorboard_path)
        exp_path = os.path.join(writer_path, 'exp_' + today)

        return SummaryWriter(exp_path)

    def run_train_epoch(self,args, global_step, n_gpu, device, model, param_optimizer, optimizer,
                        train_examples, train_features, eval_examples, eval_features,
                        logger, log_path, save_path, best_f1, epoch):

        running_loss, count = 0.0, 0
        model.train()
        for step, features in enumerate(batching.get_epoch(train_features)):
            batch_features, batch_tensors = get_tensors_list(features, True, args.gradient_accumulation_steps,
                                                             args.max_seq_length)
            for batch_feature, batch_tensor in zip(batch_features, batch_tensors):
                if n_gpu == 1:
                    batch_tensor = tuple(t.to(device) for t in batch_tensor)  # multi-gpu does scattering it-self
                input_ids, input_mask, segment_ids, number_indices, start_indices, end_indices, number_of_answers, \
                input_counts, add_sub_expressions, negations = batch_tensor

                with torch.no_grad():
                    output_dict = model("normal", input_ids, segment_ids, input_mask, number_indices)
                if len(args.answering_abilities) >= 1:
                    best_answer_ability = output_dict["best_answer_ability"]
                number_sign_logits = output_dict["number_sign_logits"]
                number_mask = output_dict["number_mask"]

                batch_result = []
                for i, feature in enumerate(batch_feature):
                    unique_id = int(feature.unique_id)
                    result = {}
                    result['unique_id'] = unique_id
                    if len(args.answering_abilities) >= 1:
                        result['predicted_ability'] = best_answer_ability[i].detach().cpu().numpy()
                    result['number_sign_logits'] = number_sign_logits[i].detach().cpu().numpy()
                    result['number_mask'] = number_mask[i].detach().cpu().numpy()
                    batch_result.append(result)

                number_indices2, sign_indices, sign_labels, _ = \
                    batch_annotate_candidates(train_examples, batch_feature, batch_result, args.answering_abilities,
                                              True,
                                              args.beam_size, args.max_count)

                number_indices2 = torch.tensor(number_indices2, dtype=torch.long)
                sign_indices = torch.tensor(sign_indices, dtype=torch.long)
                sign_labels = torch.tensor(sign_labels, dtype=torch.long)
                number_indices2 = number_indices2.to(device)
                sign_indices = sign_indices.to(device)
                sign_labels = sign_labels.to(device)

                loss = model("normal", input_ids, segment_ids, input_mask, number_indices,
                             start_indices, end_indices, number_of_answers, input_counts, add_sub_expressions,
                             negations,
                             number_indices2, sign_indices, sign_labels)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1 and len(batch_tensors) > 1:
                    loss = loss / len(batch_tensors)
                loss.backward()
                running_loss += loss.item()

            if args.fp16 or args.optimize_on_cpu:
                if args.fp16 and args.loss_scale != 1.0:
                    # scale down gradients for fp16 training
                    for param in model.parameters():
                        param.grad.data = param.grad.data / args.loss_scale
                is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                if is_nan:
                    logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                    args.loss_scale = args.loss_scale / 2
                    model.zero_grad()
                    continue
                optimizer.step()
                copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
            else:
                optimizer.step()
            model.zero_grad()
            global_step += 1
            count += 1

            if global_step % 1000 == 0 and count != 0:
                logger.info("step: {}, loss: {:.3f}".format(global_step, running_loss / count))
                running_loss, count = 0.0, 0

        logger.info("***** Running evaluation *****")
        model.eval()
        metrics = evaluate(args, model, device, eval_examples, eval_features, logger)
        f = open(log_path, "a")
        print("step: {}, em: {:.3f}, f1: {:.3f}"
              .format(global_step, metrics['em'], metrics['f1']), file=f)
        print(" ", file=f)
        f.close()
        if metrics['f1'] > best_f1:
            logger.info("Save model at {} (step {}, epoch {})".format(save_path, global_step, epoch))
            best_f1 = metrics['f1']
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': global_step,
                'epoch': epoch
            }, save_path)
        return global_step, model, best_f1
    def train(self):
        global_step, model, best_f1 = self.run_train_epoch(args, global_step, n_gpu, device, model, param_optimizer,
                                                           optimizer,
                                                           train_examples, train_features, eval_examples, eval_features,
                                                           logger, log_path, save_path, best_f1, epoch)