#!/usr/bin/env python
# encoding: utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import json
import re
import time
import logging
import random
import unicodedata
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertForMaskedLM, BertTokenizerFast

import torch
import torch.optim as optim
from tqdm import tqdm, trange
from tqdm.contrib import tzip

from bert.tokenization import BertTokenizer
import utils
from coreference import CorefModel
import conll
import metrics

format = '%(asctime)s - %(levelname)s - %(name)s.%(funcName)s:%(lineno)d - %(message)s'
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train_coref(config):
    """
    指代消解模型训练
    :param config: 配置参数
    :return: None
    """
    if config.checkpoint and config.from_checkpoint:
        try:
            model = torch.load(os.path.join(config["checkpoint"], "pytorch_model.pt"))
        except Exception as e:
            print(f"\ntrying load checkpoint from {config['checkpoint']} failed, try to load initial pretrained model!!!")
            model = CorefModel.from_pretrained(config["pretrained_model"], coref_task_config=config)
    else:
        model = CorefModel.from_pretrained(config["pretrained_model"], coref_task_config=config)
    print(model)
    logger.info("CorefModel:")
    logger.info(model)
    model.to(device)

    examples = model.get_train_example()
    train_steps = config["num_epochs"] * config["num_docs"]

    param_optimizer = list(model.named_parameters())
    print("需要学习的参数：{}".format(len(param_optimizer)))

    bert_params = list(map(id, model.bert.parameters()))
    task_params = filter(lambda p: id(p) not in bert_params, model.parameters())
    # 优化器
    optimizer = optim.Adam([
        {'params': task_params},
        {'params': model.bert.parameters(), 'lr': config['bert_learning_rate']}],
        lr=config['task_learning_rate'],
        eps=config['adam_eps'])

    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=int(train_steps * 0.1))

    logger.info("********** Running training ****************")
    logger.info("  Num train examples = %d", len(examples))
    logger.info("  Num epoch = %d", config["num_epochs"])
    logger.info("  Num train step = %d", train_steps)

    fh = logging.FileHandler(os.path.join(config["data_dir"], 'train.log'), mode="w")
    fh.setFormatter(logging.Formatter(format))
    logger.addHandler(fh)

    model.train()
    global_step = 0
    start_time = time.time()
    accumulated_loss = 0.0

    for _ in trange(int(config["num_epochs"]), desc="Epoch"):
        if config.shuffle_train:
            random.shuffle(examples)
        for step, example in enumerate(tqdm(examples, desc="Train_Examples")):
            tensorized_example = model.tensorize_example(example, is_training=True)

            input_ids = torch.from_numpy(tensorized_example["input_ids"]).long().to(device)
            input_mask = torch.from_numpy(tensorized_example["input_mask"]).long().to(device)
            text_len = torch.from_numpy(tensorized_example["text_len"]).long().to(device)
            speaker_ids = torch.from_numpy(tensorized_example["speaker_ids"]).long().to(device)
            genre = torch.tensor(tensorized_example["genre"]).long().to(device)
            is_training = tensorized_example["is_training"]
            gold_starts = torch.from_numpy(tensorized_example["gold_starts"]).long().to(device)
            gold_ends = torch.from_numpy(tensorized_example["gold_ends"]).long().to(device)
            cluster_ids = torch.from_numpy(tensorized_example["cluster_ids"]).long().to(device)
            sentence_map = torch.Tensor(tensorized_example["sentence_map"]).long().to(device)

            predictions, loss = model(input_ids, input_mask, text_len, speaker_ids, genre, is_training,
                                      gold_starts, gold_ends, cluster_ids, sentence_map)

            accumulated_loss += loss.item()
            if global_step % report_frequency == 0:
                total_time = time.time() - start_time
                steps_per_second = global_step / total_time
                average_loss = accumulated_loss / report_frequency
                print("\n")
                logger.info("step:{} | loss: {} | step/s: {}".format(global_step, average_loss, steps_per_second))
                accumulated_loss = 0.0
            # 验证集验证
            if global_step % eval_frequency == 0 and global_step != 0:
                utils.save_model(model, config["model_save_path"])
                utils.save_model_all(model, config["model_save_path"])
                torch.cuda.empty_cache()
                eval_model = CorefModel.from_pretrained(config["model_save_path"], coref_task_config=config)
                eval_model.to(device)
                eval_model.eval()
                try:
                    eval_model.evaluate(eval_model, device, official_stdout=True, eval_mode=True)
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        print("WARNING: out of memory")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise exception
                except AttributeError as exception:
                    print("Found too many repeated mentions (> 10) in the response, so refusing to score")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        scheduler.step()

    utils.save_model(model, config["model_save_path"])
    utils.save_model_all(model, config["model_save_path"])
    print("*****************************训练完成，已保存模型****************************************")
    torch.cuda.empty_cache()


def eval_coref(config):
    """
    指代消解模型验证
    :param config: 配置参数
    :return: None
    """
    model = CorefModel.from_pretrained(config["model_save_path"], coref_task_config=config)
    model.to(device)

    examples = model.get_eval_example()

    logger.info("********** Running Eval ****************")
    logger.info("  Num dev examples = %d", len(examples))

    model.eval()
    coref_predictions = {}
    coref_evaluator = metrics.CorefEvaluator()
    doc_keys = []
    keys = None
    with torch.no_grad():
        for example_num, example in enumerate(tqdm(examples, desc="Eval_Examples")):
            tensorized_example = model.tensorize_example(example, is_training=False)

            input_ids = torch.from_numpy(tensorized_example["input_ids"]).long().to(device)
            input_mask = torch.from_numpy(tensorized_example["input_mask"]).long().to(device)
            text_len = torch.from_numpy(tensorized_example["text_len"]).long().to(device)
            speaker_ids = torch.from_numpy(tensorized_example["speaker_ids"]).long().to(device)
            genre = torch.tensor(tensorized_example["genre"]).long().to(device)
            is_training = tensorized_example["is_training"]
            gold_starts = torch.from_numpy(tensorized_example["gold_starts"]).long().to(device)
            gold_ends = torch.from_numpy(tensorized_example["gold_ends"]).long().to(device)
            cluster_ids = torch.from_numpy(tensorized_example["cluster_ids"]).long().to(device)
            sentence_map = torch.Tensor(tensorized_example["sentence_map"]).long().to(device)

            if keys is not None and example['doc_key'] not in keys:
                continue
            doc_keys.append(example['doc_key'])

            (candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends,
             top_antecedents, top_antecedent_scores), loss = model(input_ids, input_mask, text_len, speaker_ids,
                                                                   genre, is_training, gold_starts, gold_ends,
                                                                   cluster_ids, sentence_map)

            predicted_antecedents = model.get_predicted_antecedents(top_antecedents.cpu(), top_antecedent_scores.cpu())
            coref_predictions[example["doc_key"]] = model.evaluate_coref(top_span_starts, top_span_ends,
                                                                         predicted_antecedents, example["clusters"],
                                                                         coref_evaluator)
    official_stdout = True
    eval_mode = True
    summary_dict = {}
    if eval_mode:
        conll_results = conll.evaluate_conll(config["conll_eval_path"], coref_predictions,
                                             model.subtoken_maps, official_stdout)
        average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
        summary_dict["Average F1 (conll)"] = average_f1
        print("Average F1 (conll): {:.2f}%".format(average_f1))

    p, r, f = coref_evaluator.get_prf()
    summary_dict["Average F1 (py)"] = f
    print("Average F1 (py): {:.2f}% on {} docs".format(f * 100, len(doc_keys)))
    summary_dict["Average precision (py)"] = p
    print("Average precision (py): {:.2f}%".format(p * 100))
    summary_dict["Average recall (py)"] = r
    print("Average recall (py): {:.2f}%".format(r * 100))


def test_coref(config):
    """
    指代消解模型预测
    :param config: 配置参数
    :return: None 
    """
    model = CorefModel.from_pretrained(config["model_save_path"], coref_task_config=config)
    model.to(device)

    output_filename = config["test_output_path"]
    examples = model.get_test_example()

    logger.info("********** Running Test ****************")
    logger.info("  Num test examples = %d", len(examples))

    model.eval()
    with open(output_filename, 'w', encoding="utf-8") as output_file:
        with torch.no_grad():
            for example_num, example in enumerate(tqdm(examples, desc="Test_Examples")):
                tensorized_example = model.tensorize_example(example, is_training=False)

                input_ids = torch.from_numpy(tensorized_example["input_ids"]).long().to(device)
                input_mask = torch.from_numpy(tensorized_example["input_mask"]).long().to(device)
                text_len = torch.from_numpy(tensorized_example["text_len"]).long().to(device)
                speaker_ids = torch.from_numpy(tensorized_example["speaker_ids"]).long().to(device)
                genre = torch.tensor(tensorized_example["genre"]).long().to(device)
                is_training = tensorized_example["is_training"]
                gold_starts = torch.from_numpy(tensorized_example["gold_starts"]).long().to(device)
                gold_ends = torch.from_numpy(tensorized_example["gold_ends"]).long().to(device)
                cluster_ids = torch.from_numpy(tensorized_example["cluster_ids"]).long().to(device)
                sentence_map = torch.Tensor(tensorized_example["sentence_map"]).long().to(device)

                (_, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores), _ = \
                    model(input_ids, input_mask, text_len, speaker_ids, genre,
                          is_training, gold_starts, gold_ends,
                          cluster_ids, sentence_map)

                predicted_antecedents = model.get_predicted_antecedents(top_antecedents.cpu(),
                                                                        top_antecedent_scores.cpu())
                example["predicted_clusters"], _ = model.get_predicted_clusters(top_span_starts, top_span_ends,
                                                                                predicted_antecedents)
                # 将句中索引——>文字
                example_sentence = utils.flatten(example["sentences"])
                predicted_list = []
                for same_entity in example["predicted_clusters"]:
                    same_entity_list = []
                    num_same_entity = len(same_entity)
                    for index in range(num_same_entity):
                        entity_name = ''.join(example_sentence[same_entity[index][0]: same_entity[index][1] + 1])
                        same_entity_list.append(entity_name)
                    predicted_list.append(same_entity_list)
                    same_entity_list = []  # 清空list

                example["predicted_idx2entity"] = predicted_list
                example["top_spans"] = list(zip((int(i) for i in top_span_starts), (int(i) for i in top_span_ends)))
                example['head_scores'] = []

                output_file.write(json.dumps(example, ensure_ascii=False))
                output_file.write("\n")
                if example_num % 100 == 0:
                    print('\n')
                    print("写入 {} examples.".format(example_num + 1))


def online_test_coref(config, input_text):
    """
    输入一段文本，进行指代消解任务
    :param config: 配置参数
    :return: None
    """

    def create_example(text):
        """将文字转为模型需要的样例格式"""
        """
        tokenized_example = tokenizer(text, return_offsets_mapping=True)
        sentences = [tokenized_example["input_ids"]]
        offset_mappings = [tokenized_example["offset_mapping"]]
        """
        """
        sentences = [['[CLS]'] + tokenizer.tokenize_not_UNK(text) + ['[SEP]']]
        tokens = sentences
        sentences = [tokenizer.convert_tokens_to_ids(sentences[0])]
        """

        tokenized_example = tokenizer.encode_not_UNK(text=text, add_special_tokens=True, output_offset_mapping=True)
        sentences = [tokenized_example["input_ids"]]
        tokens = [['[CLS]'] + tokenizer.tokenize_not_UNK(text) + ['[SEP]']]
        offset_mappings = [tokenized_example["offset_mapping"]]

        sentence_map = [0] * len(sentences[0])
        speakers = [["-" for _ in sentence] for sentence in sentences]
        subtoken_map = [i for i in range(len(sentences[0]))]
        return {
            "doc_key": "bn",
            "clusters": [],
            "sentences": sentences,
            "text": [text],
            "tokens": tokens,
            "speakers": speakers,
            'sentence_map': sentence_map,
            'subtoken_map': subtoken_map,
            "offset_mappings": offset_mappings
        }

    tokenizer = BertTokenizer.from_pretrained(config['vocab_file'], do_lower_case=True)
    #tokenizer = AutoTokenizer.from_pretrained(config['pretrained_model'], do_lower_case=True)

    online_coref_output_file = config['online_output_path']

    example = create_example(input_text)

    model = CorefModel.from_pretrained(config["model_save_path"], coref_task_config=config)
    #model = torch.load(os.path.join(config["model_save_path"], "pytorch_model.pt"))
    model.to(device)

    model.eval()
    with open(online_coref_output_file, 'w', encoding="utf-8") as output_file:

        with torch.no_grad():
            tensorized_example = model.tensorize_example(example, is_training=False, is_predict=True)

            input_ids = torch.from_numpy(tensorized_example["input_ids"]).long().to(device)
            input_mask = torch.from_numpy(tensorized_example["input_mask"]).long().to(device)
            text_len = torch.from_numpy(tensorized_example["text_len"]).long().to(device)
            speaker_ids = torch.from_numpy(tensorized_example["speaker_ids"]).long().to(device)
            genre = torch.tensor(tensorized_example["genre"]).long().to(device)
            is_training = tensorized_example["is_training"]
            gold_starts = torch.from_numpy(tensorized_example["gold_starts"]).long().to(device)
            gold_ends = torch.from_numpy(tensorized_example["gold_ends"]).long().to(device)
            cluster_ids = torch.from_numpy(tensorized_example["cluster_ids"]).long().to(device)
            sentence_map = torch.Tensor(tensorized_example["sentence_map"]).long().to(device)

            (_, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores), _ = \
                model(input_ids, input_mask, text_len, speaker_ids, genre,
                      is_training, gold_starts, gold_ends,
                      cluster_ids, sentence_map)

            predicted_antecedents = model.get_predicted_antecedents(top_antecedents.cpu(),
                                                                    top_antecedent_scores.cpu())
            # 预测实体索引
            example["predicted_clusters"], _ = model.get_predicted_clusters(top_span_starts, top_span_ends,
                                                                            predicted_antecedents)
            # 索引——>文字
            """
            example_sentence = utils.flatten(example["tokens"])
            """

            example_sentence = example["text"][0]
            example["predicted_clusters"] = get_spans_char_level(example["predicted_clusters"], example["offset_mappings"][0])

            predicted_list = []
            for same_entity in example["predicted_clusters"]:
                same_entity_list = []
                num_same_entity = len(same_entity)
                for index in range(num_same_entity):
                    """
                    entity_name = ''.join(example_sentence[same_entity[index][0]: same_entity[index][1] + 1])
                    same_entity_list.append(entity_name)
                    """
                    """
                    tokens = example_sentence[same_entity[index][0]: same_entity[index][1] + 1]
                    #tokens = [token[2:] if re.fullmatch("^##\w+", token) else token for token in tokens]
                    entity_name = ''.join(tokens)
                    same_entity_list.append(entity_name)
                    """

                    entity_name = example_sentence[same_entity[index][0]: same_entity[index][1]]
                    same_entity_list.append(entity_name)

                predicted_list.append(same_entity_list)
                same_entity_list = []  # 清空list
            example["predicted_idx2entity"] = predicted_list

            example["top_spans"] = list(zip((int(i) for i in top_span_starts), (int(i) for i in top_span_ends)))
            example['head_scores'] = []

            output_file.write(json.dumps(example, ensure_ascii=False))
            output_file.write("\n")
    return example

def get_spans_char_level(spans, offset_map):
    sentence_id = []
    for span in spans:
        cluster_id = []
        for start, end in span:
            cluster_id.append((offset_map[start][0], offset_map[end][1]))
        cluster_id = tuple(cluster_id)
        sentence_id.append(cluster_id)
    return sentence_id


if __name__ == "__main__":

    os.environ["data_dir"] = "./data"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="bert-base-chinese", type=str, help="base pretrained model")
    parser.add_argument("--from_checkpoint", action="store_true")
    parser.add_argument("--shuffle_train", action="store_true")
    args = parser.parse_args()

    # run_experiment = "bert_base_chinese"
    # run_experiment = "roberta_L6_H768"
    # run_experiment = "chinese-lert-base"
    run_experiment = args.config
    config = utils.read_config(run_experiment, "experiments.conf")
    config["from_checkpoint"] = args.from_checkpoint
    config["shuffle_train"] = args.shuffle_train
    report_frequency = config["report_frequency"]
    eval_frequency = config["eval_frequency"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(41)
    else:
        torch.manual_seed(41)

    # 训练阶段
    if config["do_train"]:
        train_coref(config)

    # 验证阶段
    if config["do_eval"]:
        try:
            eval_coref(config)
        except AttributeError as exception:
            print("Found too many repeated mentions (> 10) in the response, so refusing to score")

    # 测试阶段
    if config["do_test"]:
        test_coref(config)

    # 单句样本测试
    if config["do_one_example_test"]:
        """
        # with open("data/predict/text.txt", 'r', encoding='utf8') as f:
        #     texts = f.readlines()
        # texts = [text.strip().replace('“', '"').replace('”', '"').replace('…', '...').replace('—', '-') for text in texts]
        # texts = [unicodedata.normalize("NFKC", text) for text in texts]
        # # input_text = "我的偶像是姚明，他喜欢打篮球，他的老婆叫叶莉。"
        # # input_text = "百邦科技：达安世纪协议转让约651万股完成过户 百邦科技(SZ 300736，收盘价：10.08元)7月25日晚间发布公告称，达安世纪于2022年7月2日与刘一苇先生签署了《股份转让协议》，拟将其持有的约651万股公司无限售条件流通股转让给刘一苇先生，占公司总股本5.15%，转让价格为8.1元/股，转让价款总计人民币约5270万元。中国证券登记结算有限责任公司深圳分公司出具的《证券过户登记确认书》。 2021年1至12月份，百邦科技的营业收入构成为：居民服务和修理及其他服务业占比97.18%。 百邦科技的总经理、董事长均是刘铁峰，男，50岁，学历背景为硕士。 截至发稿，百邦科技市值为13亿元。"
        # input_text = "黎明， 性别男，汉族，中国公民，他在1987年6月出生在山东省聊城市。"
        # 
        # for i, text in enumerate(tqdm(texts)):
        #     text = text[0: config["max_segment_len"] - 2]
        #     config['online_output_path'] = f"data/predict/{i}.jsonl"
        #     online_test_coref(config, text)
        """

        examples = utils.load_json_data("data/predict/v2/coreference.json")
        texts = [example["query"] for example in examples]
        texts = [text.strip().replace('“', '"').replace('”', '"').replace('…', '...').replace('—', '-') for text in texts]
        texts = [unicodedata.normalize("NFKC", text) for text in texts]

        for text, example in tzip(texts, examples):
            text = text[0: config["max_segment_len"] - 2]
            result = online_test_coref(config, text)
            example["coref"] = {"text": result["text"][0],
                                "clusters": result["predicted_clusters"],
                                "clusters_strings": result["predicted_idx2entity"]}
            utils.save2json(examples, save_path="data/predict/v2/coref_results.json")


