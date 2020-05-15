from __future__ import absolute_import

from .resnet import *

from .nasnet import *
from .mobilenetv2 import *
from .shufflenet import *
from .squeezenet import *
from .shufflenetv2 import shufflenetv2

__model_factory = {
    # image classification models
    'resnet50': resnet50,
    'resnet50_fc512': resnet50_fc512,
    # lightweight models
    'nasnsetmobile': nasnetamobile,
    'mobilenetv2': MobileNetV2,
    'shufflenet': shufflenet,
    'squeezenet1_0': squeezenet1_0,
    'squeezenet1_0_fc512': squeezenet1_0_fc512,
    'squeezenet1_1': squeezenet1_1,
    'shufflenetv2': shufflenetv2
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError('Unknown model: {}'.format(name))
    return __model_factory[name](*args, **kwargs)
