from __future__ import absolute_import

from .ResNet import *
from .DenseNet import *


__factory = {
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'densenet121': DenseNet121,
    'densenet169':DenseNet169,
    'CNN': CNN,
    'CNN1':CNN1,
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)