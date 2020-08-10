from dataclasses import dataclass, field
from typing import List, Any

from hydra.core.config_store import ConfigStore

from deepself.models.nn_models.cnn import CNNConfig
from deepself.models.nn_models.cnn_rnn import CNNRNNConfig
from deepself.models.nn_models.nn import NNConfig
from deepself.models.nn_models.panns_cnn14 import PANNsConfig
from deepself.models.nn_models.pretrained_models import PretrainedConfig
from deepself.models.nn_models.rnn import RNNConfig
from deepself.tasks.base_experiment import BaseExptConfig
from deepself.utils.nn_config import SGDConfig, AdamConfig

nn_model_list = [('nn', NNConfig), ('cnn', CNNConfig), ('rnn', RNNConfig), ('cnn_rnn', CNNRNNConfig)]
nn_model_list.extend([(model_name, CNNConfig) for model_name in ['logmel_cnn', 'attention_cnn', 'panns', '1dcnn_rnn']])
pretrained_models = ['resnet', 'resnet152', 'alexnet', 'wideresnet', 'resnext', 'resnext101', 'vgg19', 'vgg16',
                     'googlenet', 'mobilenet', 'panns', 'resnext_wsl']
pretrained_model_list = [(model_name, PretrainedConfig) for model_name in pretrained_models]
extended_models = [('panns', PANNsConfig)]
model_list = nn_model_list + pretrained_model_list + extended_models


defaults = [
    {'train.model': 'cnn'},
    {'train.model.optim': 'adam'},
]


@dataclass
class ExptConfig(BaseExptConfig):
    defaults: List[Any] = field(default_factory=lambda: defaults)


def before_hydra(config_class):
    cs = ConfigStore.instance()
    cs.store(name='config', node=config_class)
    [cs.store(group='train.model', name=model_name, node=model_cfg) for model_name, model_cfg in model_list]
    cs.store(group='train.model.optim', name='sgd', node=SGDConfig)
    cs.store(group='train.model.optim', name='adam', node=AdamConfig)
    return cs
