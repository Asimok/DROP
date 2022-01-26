import json
import os
import random
from datetime import datetime

import numpy as np
import torch
from pytorch_pretrained_bert import BertConfig
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from dataset_dataloaders.drop_dataloader import DropDataloader, my_collate
from drop.drop_metric import DropEmAndF1
from model.drop_model import DROP_Model
from model.optimization import BERTAdam
from tools.drop_utils import batch_annotate_candidates, write_predictions
from tools.log import get_logger


class Trainer(object):
    def __init__(self, hparams, mode='train'):
        self.hparams = hparams
        self.make_dir()
        self.log = get_logger(log_name="Trainer")
        self.mode = mode
        self.device = self.hparams.device
        self.model = None

        self.summery_writer = self.init_SummaryWriter()
        self.pretrained_model_config, self.tokenizer = self.Load_pretrained_model_config()
        # TODO 加载数据根据 mode='train' 参数 减少内存占用
        self.train_dataloader,self.train_examples, self.t_total = self.build_dataloader_for_train(
            file_path=os.path.join(self.hparams.datasetPath, self.hparams.trainFile))
        self.test_dataloader,self.test_examples = self.build_dataloader_for_eval(
            file_path=os.path.join(self.hparams.datasetPath, self.hparams.testFile))


        self.criterion, self.optimizer = self.build_model()
        self.save_train_config()

    def make_dir(self):
        model_path = os.path.join(self.hparams.output_dir, self.hparams.current_model)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        log_path = os.path.join(model_path, 'logs')
        saved_model_path = os.path.join(model_path, 'saved_model')
        tensorboard_runs_path = os.path.join(model_path, 'tensorboard_runs')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if not os.path.exists(saved_model_path):
            os.makedirs(saved_model_path)
        if not os.path.exists(tensorboard_runs_path):
            os.makedirs(tensorboard_runs_path)

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

    def bert_load_state_dict(self, model, state_dict):
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

    def build_dataloader_for_train(self, file_path=None):
        self.log.info("Load main dataset from file %s ...", file_path)

        train_dataset = DropDataloader(hparams=self.hparams, evaluate=False, tokenizer=self.tokenizer,
                                       file_path=file_path)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size_for_train,
            num_workers=self.hparams.workers,
            shuffle=True,
            drop_last=True,
            collate_fn=my_collate
        )
        train_examples=train_dataset.get_examples()
        t_total = len(train_dataloader) // self.hparams.batch_size_for_train * self.hparams.train_epochs
        self.log.info("Num steps = %d", t_total)
        self.log.info("Load train dataset finished!")
        return train_dataloader,train_examples, t_total

    def build_dataloader_for_eval(self, file_path=None):
        self.log.info("Load eval dataset from file %s ...", file_path)
        test_dataset = DropDataloader(hparams=self.hparams, evaluate=True, tokenizer=self.tokenizer,
                                      file_path=file_path)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.hparams.batch_size_for_test,
            num_workers=self.hparams.workers,
            shuffle=True,
            drop_last=True,
            collate_fn=my_collate
        )
        test_examples=test_dataset.get_examples()
        self.log.info("Load eval dataset finished!")
        return test_dataloader,test_examples

    def build_model(self):
        # Define model
        self.log.info("Define model...")
        random.seed(self.hparams.seed)
        np.random.seed(self.hparams.seed)
        torch.manual_seed(self.hparams.seed)

        # self.pretrained_model_config
        bert_config_file = os.path.join(self.hparams.pretrainedModelPath, 'config.json')
        bert_config = BertConfig.from_json_file(bert_config_file)
        self.model = DROP_Model(bert_config=bert_config,
                                config=self.hparams,
                                )
        save_path = os.path.join(self.hparams.output_dir, self.hparams.best_model_save_path)
        bert_path = os.path.join(self.hparams.pretrainedModelPath, 'pytorch_model.bin')
        if bert_path is not None and not os.path.isfile(save_path):
            self.log.info("Loading model from pretrained checkpoint: {}".format(bert_path))
            self.model = self.bert_load_state_dict(self.model, torch.load(bert_path, map_location='cpu'))

        # GPU or CPU
        if not torch.cuda.is_available():
            self.device = 'cpu'
        self.log.info('use %s to train', self.device)
        self.model.to(self.device)

        # Use Multi-GPUs
        if len(self.hparams.gpu_ids) > 1 and self.device != 'cpu':
            self.model = nn.DataParallel(self.model, device_ids=self.hparams.gpu_ids)
            self.log.info("Use Multi-GPUs" + str(self.hparams.gpu_ids))
        else:
            self.log.info("Use 1 GPU")

        # Define Loss and Optimizer
        no_decay = ['bias', 'gamma', 'beta']
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
        model_path = os.path.join(self.hparams.output_dir, self.hparams.current_model)
        config_path = os.path.join(model_path, self.hparams.model_config_file_path)
        self.log.info('output_dir: {}'.format(config_path))

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
        model_path = os.path.join(self.hparams.output_dir, self.hparams.current_model)
        save_path = os.path.join(model_path, self.hparams.best_model_save_path)  # 最优模型保存路径模型
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
        model_path = os.path.join(self.hparams.output_dir, self.hparams.current_model)
        writer_path = os.path.join(model_path, self.hparams.tensorboard_path)
        exp_path = os.path.join(writer_path, 'exp_' + today)
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        return SummaryWriter(exp_path)

    # def evaluate(self, write_pred=False, batch_tensors=None, examples=None, features=None):
    def evaluate(self, write_pred=False, test_dataloader=None):
        drop_metrics = DropEmAndF1()
        all_results = []
        all_features = []
        for step, (batch_test_tensors, batch_test_features) in enumerate(test_dataloader):
            all_features+=batch_test_features
            best_answer_ability = None
            best_count_number = None
            best_negations_for_numbers = None
            if len(all_results) % 1000 == 0:
                self.log.info("Processing example: %d" % (len(all_results)))

            batch_test_tensors = tuple(t.to(self.hparams.device) for t in batch_test_tensors)
            input_ids, input_mask, segment_ids, number_indices = batch_test_tensors
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
            for i, feature in enumerate(batch_test_features):
                unique_id = int(feature['unique_id'])
                result = {'unique_id': unique_id}
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
                batch_annotate_candidates(all_examples=self.test_examples, batch_features=batch_test_features,
                                          batch_results=batch_result,
                                          answering_abilities=self.hparams.answering_abilities,
                                          is_training=False, beam_size=self.hparams.beam_size,
                                          max_count=self.hparams.max_count)
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

        all_predictions, metrics = write_predictions(self.test_examples, all_features, all_results,
                                                     self.hparams.answering_abilities, drop_metrics,
                                                     self.hparams.length_heuristic,
                                                     self.hparams.n_best_size, self.hparams.max_answer_length,
                                                     self.hparams.do_lower_case, self.hparams.verbose_logging, self.log)
        if write_pred:
            model_path = os.path.join(self.hparams.output_dir, self.hparams.current_model)
            output_prediction_file = os.path.join(model_path, self.hparams.prediction_file_path)
            with open(output_prediction_file, "w") as writer:
                writer.write(json.dumps(all_predictions, indent=4) + "\n")
            self.log.info("Writing predictions to: %s" % output_prediction_file)

        return metrics

    def run_train_epoch(self, n_gpu, device, global_step, best_f1, epoch):

        # self.log.info("***** Prepare for Train *****")
        train_begin = datetime.utcnow()  # Times
        global_iteration_step = 0
        # Train
        # TODO main model
        # self.log.info("Num Epochs = %d", int(self.hparams.train_epochs))
        # self.log.info("batch size = %d", int(self.hparams.batch_size_for_train))
        total_epoch = int(self.hparams.train_epochs)
        self.model.train()
        running_loss, count = 0.0, 0
        all_examples=[]
        bar_format = '{desc}{percentage:2.0f}%|{bar}|{n_fmt}/{total_fmt}[{elapsed}<{remaining}{postfix}]'
        epoch_iterator = tqdm(self.train_dataloader, ncols=120, bar_format=bar_format)
        epoch_iterator.set_description('Epoch: {}/{}'.format(epoch, total_epoch))  # 设置前缀 一般为epoch的信息
        for step, (batch_train_tensor, batch_train_features) in enumerate(epoch_iterator):
            best_answer_ability = None
            if n_gpu == 1:
                batch_train_tensor = tuple(
                    t.to(self.hparams.device) for t in batch_train_tensor)  # multi-gpu does scattering it-self
            input_ids, input_mask, segment_ids, number_indices, start_indices, end_indices, number_of_answers, \
            input_counts, add_sub_expressions, negations = batch_train_tensor

            with torch.no_grad():
                output_dict = self.model("normal", input_ids, segment_ids, input_mask, number_indices)
            if len(self.hparams.answering_abilities) >= 1:
                best_answer_ability = output_dict["best_answer_ability"]
            number_sign_logits = output_dict["number_sign_logits"]
            number_mask = output_dict["number_mask"]

            batch_result = []
            for i, feature in enumerate(batch_train_features):
                unique_id = int(feature['unique_id'])
                # unique_id = feature['unique_id']
                result = {'unique_id': unique_id}
                if len(self.hparams.answering_abilities) >= 1:
                    result['predicted_ability'] = best_answer_ability[i].detach().cpu().numpy()
                result['number_sign_logits'] = number_sign_logits[i].detach().cpu().numpy()
                result['number_mask'] = number_mask[i].detach().cpu().numpy()
                batch_result.append(result)


            number_indices2, sign_indices, sign_labels, _ = \
                batch_annotate_candidates(all_examples=self.train_examples, batch_features=batch_train_features,
                                          batch_results=batch_result,
                                          answering_abilities=self.hparams.answering_abilities,
                                          is_training=True,
                                          beam_size=self.hparams.beam_size, max_count=self.hparams.max_count)

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

            if self.hparams.gradient_accumulation_steps > 1 and len(batch_train_tensor) > 1:
                loss = loss / len(batch_train_tensor)
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
        # evaluate
        metrics = self.evaluate(write_pred=True, test_dataloader=self.test_dataloader)
        model_path = os.path.join(self.hparams.output_dir, self.hparams.current_model)
        performance_file_path = os.path.join(model_path, self.hparams.performance_file_path)
        f = open(performance_file_path, "a")
        print("step: {}, em: {:.3f}, f1: {:.3f}"
              .format(global_step, metrics['em'], metrics['f1']), file=f)
        print(" ", file=f)
        f.close()
        self.log.info('save performance_file to : ' + performance_file_path)
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            self.save_best_model(model=self.model, optimizer=self.optimizer,
                                 global_step=global_step, epoch=epoch)

        return global_step, self.model, best_f1

    def run_predict(self):
        # --- Run prediction ---
        self.log.info("***** Running prediction *****")
        # restore from best checkpoint

        global_step = 0
        model_path = os.path.join(self.hparams.output_dir, self.hparams.current_model)
        save_path = os.path.join(model_path, self.hparams.best_model_save_path)  # 最优模型保存路径模型
        if save_path and os.path.isfile(save_path):
            checkpoint = torch.load(save_path)
            self.model.load_state_dict(checkpoint['model'])
            self.log.info("Loading model from fine-tuned checkpoint: '{}' (step {}, epoch {})"
                          .format(save_path, checkpoint['step'], checkpoint['epoch']))
            global_step = checkpoint['step']

            torch.save({
                'model': self.model.state_dict(),
                'step': checkpoint['step'],
                'epoch': checkpoint['epoch']
            }, save_path)

        self.model.eval()
        metrics = self.evaluate(write_pred=True, test_dataloader=self.test_dataloader)
        model_path = os.path.join(self.hparams.output_dir, self.hparams.current_model)
        performance_file_path = os.path.join(model_path, self.hparams.performance_file_path)
        f = open(performance_file_path, "a")
        print("predict step: {}, em: {:.3f}, f1: {:.3f}"
              .format(global_step, metrics['em'], metrics['f1']), file=f)
        print(" ", file=f)
        f.close()

        self.log.info("predict step: {}, test_em: {:.3f}, test_f1: {:.3f}"
                      .format(global_step, metrics['em'], metrics['f1']))
