from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .satudora import SATUDORA

__imgreid_factory = {
    'satudora':SATUDORA
}

def init_imgreid_dataset(name, **kwargs):
    if name not in list(__imgreid_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__imgreid_factory.keys())))
    return __imgreid_factory[name](**kwargs)