import json

import torch
from torch import nn
import pyhocon, os, errno


def to_cuda(x):
    """ GPU-enable a tensor """
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def flatten(l):
    """展平list"""
    return [item for sublist in l for item in sublist]


def read_config(run_experiment, file_name):
    """读取配置文件"""
    name = str(run_experiment)
    print("Running experiment: {}".format(name))
    config = pyhocon.ConfigFactory.parse_file(file_name)[name]

    print(pyhocon.HOCONConverter.convert(config, "hocon"))
    return config


def save_model(model, save_path):
    """模型保存"""
    model_to_save = model.module if hasattr(model, 'module') else model
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    output_model_file = os.path.join(save_path, "pytorch_model.bin")
    output_config_file = os.path.join(save_path, "config.json")

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.bert_config.to_json_file(output_config_file)
    model_to_save.tokenizer.save_vocabulary(save_path)


def save_model_all(model, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_save_path = os.path.join(save_path, "pytorch_model.pt")
    torch.save(model, model_save_path)


def save2json(obj, save_path, indent=None):
    with open(save_path, 'w', encoding='utf8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)

def load_json_data(data_path):
    with open(data_path, 'r', encoding='utf8') as f:
        return json.load(f)
