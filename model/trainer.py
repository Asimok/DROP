import json
import os
from datetime import datetime

import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from config.hparams import PARAMS
from dataset_dataloaders.drop_dataloader import DropDataloader
from drop.drop_metric import DropEmAndF1
from model.drop_model import DROP_Model
from model.optimization import BERTAdam
from tools.drop_utils import batch_annotate_candidates, write_predictions
from tools.log import get_logger


class Trainer(object):
    def __init__(self, hparams, mode='main'):
        self.log = get_logger(log_name="Trainer")
        self.hparams = hparams
        self.mode = mode
        self.device = self.hparams.device
        self.model = None

        self.summery_writer = self.init_SummaryWriter()
        self.pretrained_model_config, self.tokenizer = self.Load_pretrained_model_config()
        # TODO 加载数据根据 mode='main' 参数 减少内存占用
        self.train_dataloader, self.train_examples, self.t_total = self.build_dataloader_for_train()
        self.test_dataloader, self.test_examples = self.build_dataloader_for_test()
        self.criterion, self.optimizer = self.build_model()
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

    def bert_load_state_dict(self,model, state_dict):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')

        if len(missing_keys) > 0:
            self.log.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            self.log.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        return model

    def build_dataloader_for_train(self):
        self.log.info("Load main dataset from file %s ...",
                      os.path.join(self.hparams.datasetPath, self.hparams.trainFile))

        train_dataset = DropDataloader(hparams=self.hparams, evaluate=False, tokenizer=self.tokenizer)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size_for_train,
            num_workers=self.hparams.workers,
            shuffle=True,
            drop_last=True
        )
        train_examples = train_dataset.get_examples()
        t_total = len(train_dataloader) // self.hparams.batch_size_for_train * self.hparams.train_epochs
        self.log.info("Num steps = %d", t_total)
        self.log.info("Load main dataset finished!!!")
        return train_dataloader, train_examples, t_total

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
        self.log.info("Load main dataset finished!!!")
        return test_dataloader, test_examples

    def build_model(self):
        # Define model
        self.log.info("Define model...")
        self.model = DROP_Model(bert_config=self.pretrained_model_config,
                                config=self.hparams,
                                )
        save_path = os.path.join(self.hparams.output_dir, self.hparams.best_model_save_path)
        bert_path =os.path.join(self.hparams.pretrainedModelPath,'pytorch_model.bin')
        if bert_path is not None and not os.path.isfile(save_path):
            self.log.info("Loading model from pretrained checkpoint: {}".format(bert_path))
            self.model = self.bert_load_state_dict(self.model, torch.load(bert_path, map_location='cpu'))


        # GPU or CPU
        if not torch.cuda.is_available():
            self.device = 'cpu'
        self.log.info('use =%s to main', self.device)
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
            config_fields = self.hparams
            for k, v in config_fields._asdict().items():
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
        save_path = os.path.join(self.hparams.output_dir, self.hparams.best_model_save_path)  # 最优模型保存路径模型
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

    def evaluate(self):

        drop_metrics = DropEmAndF1()
        all_results = []
        for step, batch in enumerate(self.test_dataloader):
            best_answer_ability = None
            best_count_number = None
            best_negations_for_numbers = None
            if len(all_results) % 1000 == 0:
                self.log.info("Processing example: %d" % (len(all_results)))

            batch = tuple(t.to(self.hparams.device) for t in batch)
            input_ids, input_mask, segment_ids, number_indices = batch
            with torch.no_grad():
                output_dict = self.model("normal", input_ids, segment_ids, input_mask, number_indices)

            if len(self.hparams.answering_abilities) >= 1:
                best_answer_ability = output_dict["best_answer_ability"]
            span_start_logits = output_dict["span_start_logits"]
            span_end_logits = output_dict["span_end_logits"]
            best_span_number = output_dict["best_span_number"]
            number_sign_logits = output_dict["number_sign_logits"]
            number_mask = output_dict["number_mask"]
            encoded_numbers_output = output_dict["encoded_numbers_output"]
            passage_output = output_dict["passage_output"]
            question_output = output_dict["question_output"]
            pooled_output = output_dict["pooled_output"]
            if "counting" in self.hparams.answering_abilities:
                best_count_number = output_dict["best_count_number"]
            if "negation" in self.hparams.answering_abilities:
                best_negations_for_numbers = output_dict["best_negations_for_numbers"]

            batch_result = []
            for i, feature in enumerate(self.test_examples):
                unique_id = int(feature.unique_id)
                result = {}
                result['unique_id'] = unique_id
                if len(self.hparams.answering_abilities) >= 1:
                    result['predicted_ability'] = best_answer_ability[i].detach().cpu().numpy()
                result['start_logits'] = span_start_logits[i].detach().cpu().tolist()
                result['end_logits'] = span_end_logits[i].detach().cpu().tolist()
                result['predicted_spans'] = best_span_number[i].detach().cpu().numpy()
                result['number_sign_logits'] = number_sign_logits[i].detach().cpu().numpy()
                result['number_mask'] = number_mask[i].detach().cpu().numpy()
                if "counting" in self.hparams.answering_abilities:
                    result['predicted_count'] = best_count_number[i].detach().cpu().numpy()
                if "negation" in self.hparams.answering_abilities:
                    result['predicted_negations'] = best_negations_for_numbers[i].detach().cpu().numpy()
                batch_result.append(result)

            number_indices2, sign_indices, _, sign_scores = \
                batch_annotate_candidates(self.test_examples, batch, batch_result, self.hparams.answering_abilities,
                                          False, self.hparams.beam_size, self.hparams.max_count)
            number_indices2 = torch.tensor(number_indices2, dtype=torch.long)
            sign_indices = torch.tensor(sign_indices, dtype=torch.long)
            number_indices2 = number_indices2.to(self.hparams.device)
            sign_indices = sign_indices.to(self.hparams.device)

            with torch.no_grad():
                sign_rerank_logits = self.model("rerank_inference", input_ids, segment_ids, input_mask, number_indices,
                                                number_indices2=number_indices2, sign_indices=sign_indices,
                                                encoded_numbers_input=encoded_numbers_output,
                                                passage_input=passage_output,
                                                question_input=question_output, pooled_input=pooled_output)

            for i, result in enumerate(batch_result):
                result['number_indices2'] = number_indices2[i].detach().cpu().tolist()
                result['sign_indices'] = sign_indices[i].detach().cpu().tolist()
                result['sign_rerank_logits'] = sign_rerank_logits[i].detach().cpu().tolist()
                result['sign_probs'] = sign_scores[i]
                all_results.append(result)

        all_predictions, metrics = write_predictions(self.test_examples, self.test_dataloader, all_results,
                                                     self.hparams.answering_abilities, drop_metrics,
                                                     self.hparams.length_heuristic,
                                                     self.hparams.n_best_size, self.hparams.max_answer_length,
                                                     self.hparams.do_lower_case, self.hparams.verbose_logging, self.log)

        output_prediction_file = os.path.join(self.hparams.output_dir, "predictions.json")
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
        self.log.info("Writing predictions to: %s" % output_prediction_file)

        return metrics

    def run_train_epoch(self, n_gpu, device, global_step, best_f1, epoch):

        self.log.info("***** Prepare for Train *****")
        train_begin = datetime.utcnow()  # Times
        global_iteration_step = 0
        # Train
        # TODO main model
        self.log.info("Num Epochs = %d", int(self.hparams.train_epochs))
        self.log.info("batch size = %d", int(self.hparams.batch_size))
        total_epoch = int(self.hparams.train_epochs)
        self.model.train()
        running_loss, count = 0.0, 0
        bar_format = '{desc}{percentage:2.0f}%|{bar}|{n_fmt}/{total_fmt}[{elapsed}<{remaining}{postfix}]'
        epoch_iterator = tqdm(self.train_dataloader, ncols=120, bar_format=bar_format)
        epoch_iterator.set_description('Epoch: {}/{}'.format(epoch + 1, total_epoch))  # 设置前缀 一般为epoch的信息
        for step, batch_tensor in enumerate(epoch_iterator):
            best_answer_ability = None
            if n_gpu == 1:
                batch_tensor = tuple(
                    t.to(self.hparams.device) for t in batch_tensor)  # multi-gpu does scattering it-self
            input_ids, input_mask, segment_ids, number_indices, start_indices, end_indices, number_of_answers, \
            input_counts, add_sub_expressions, negations = batch_tensor

            with torch.no_grad():
                output_dict = self.model("normal", input_ids, segment_ids, input_mask, number_indices)
            if len(self.hparams.answering_abilities) >= 1:
                best_answer_ability = output_dict["best_answer_ability"]
            number_sign_logits = output_dict["number_sign_logits"]
            number_mask = output_dict["number_mask"]

            batch_result = []
            for i, feature in enumerate(self.train_examples):
                unique_id = int(feature['unique_id'])
                result = {}
                result['unique_id'] = unique_id
                if len(self.hparams.answering_abilities) >= 1:
                    result['predicted_ability'] = best_answer_ability[i].detach().cpu().numpy()
                result['number_sign_logits'] = number_sign_logits[i].detach().cpu().numpy()
                result['number_mask'] = number_mask[i].detach().cpu().numpy()
                batch_result.append(result)

            number_indices2, sign_indices, sign_labels, _ = \
                batch_annotate_candidates(self.train_examples, self.batch_tensor, batch_result,
                                          self.hparams.answering_abilities,
                                          True,
                                          self.hparams.beam_size, self.hparams.max_count)

            number_indices2 = torch.tensor(number_indices2, dtype=torch.long)
            sign_indices = torch.tensor(sign_indices, dtype=torch.long)
            sign_labels = torch.tensor(sign_labels, dtype=torch.long)
            number_indices2 = number_indices2.to(device)
            sign_indices = sign_indices.to(device)
            sign_labels = sign_labels.to(device)

            loss = self.model("normal", input_ids, segment_ids, input_mask, number_indices,
                              start_indices, end_indices, number_of_answers, input_counts, add_sub_expressions,
                              negations,
                              number_indices2, sign_indices, sign_labels)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            if self.hparams.gradient_accumulation_steps > 1 and len(batch_tensor) > 1:
                loss = loss / len(batch_tensor)
            loss.backward()
            running_loss += loss.item()
            self.optimizer.step()

            self.model.zero_grad()
            global_step += 1
            count += 1

            if global_step % 1000 == 0 and count != 0:
                self.log.info("step: {}, loss: {:.3f}".format(global_step, running_loss / count))
                running_loss, count = 0.0, 0

        self.log.info("***** Running evaluation *****")
        self.model.eval()
        metrics = self.evaluate()
        log_path = os.path.join(self.hparams.output_dir, self.hparams.performance)
        f = open(log_path, "a")
        print("step: {}, em: {:.3f}, f1: {:.3f}"
              .format(global_step, metrics['em'], metrics['f1']), file=f)
        print(" ", file=f)
        f.close()
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            self.save_best_model(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                                 global_step=global_step, epoch=epoch)

        return global_step, self.model, best_f1
