import os
import pickle
import numpy as np
import torch
import torchreid
import warnings
import cv2
from PIL import Image
from functools import partial
from collections import OrderedDict
from scipy.spatial.distance import cdist
from torchvision.transforms import *


try:
    import accimage
except ImportError:
    accimage = None


class Recognition(object):

    def __init__(self, device, model_path, gallery_path, gallery_id_path, top_k=1):
        self.gallery, self.gallery_id = self.load_gallery(gallery_path=gallery_path, gallery_id_path=gallery_id_path)
        self._device = device
        self._model_path = model_path
        self._top_k = top_k
        self.model = torchreid.models.build_model(
            name='osnet_ain_x1_0',
            num_classes=43,
            loss='softmax',
            pretrained=True

            # name='shufflenet_v2_x1_0',
            # num_classes=48,
            # loss='softmax',
        )
        self.model = self.model.cuda(self._device)
        self.load_pretrained_weights(self.model, self._model_path)
        self.model.eval()
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
        normalize = Normalize(mean=norm_mean, std=norm_std)
        self.transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            normalize,
        ])

    def load_pretrained_weights(self, model, weight_path):
        checkpoint = load_checkpoint(weight_path)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        model_dict = model.state_dict()
        new_state_dict = OrderedDict()
        matched_layers, discarded_layers = [], []
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]  # discard module.
            if k in model_dict and model_dict[k].size() == v.size():
                new_state_dict[k] = v
                matched_layers.append(k)
            else:
                discarded_layers.append(k)
        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict)
        if len(matched_layers) == 0:
            warnings.warn(
                'The pretrained weights "{}" cannot be loaded, '
                'please check the key names manually '
                '(** ignored and continue **)'.format(weight_path)
            )
        else:
            print(
                'Successfully loaded pretrained weights from "{}"'.
                    format(weight_path)
            )
            if len(discarded_layers) > 0:
                print(
                    '** The following layers are discarded '
                    'due to unmatched keys or layer size: {}'.
                        format(discarded_layers)
                )

    def load_gallery(self, gallery_path, gallery_id_path):
        with open(gallery_path, 'rb') as f:
            gallery = pickle.load(f)
        with open(gallery_id_path, 'rb') as f:
            gallery_id = pickle.load(f)
        return gallery, gallery_id

    def extract_feature(self, images):
        images = images.cuda(self._device)
        output = self.model(images)
        index = output.data.cpu()
        return index.numpy()

    def getOutput(self, image, bounding_box):
        if len(bounding_box) == 0:
            return []
        image = cv2_PIL(image)
        imgs = []

        for box in bounding_box:
            xmin, ymin, xmax, ymax, score = box
            crop_image = crop(image, xmin, ymin, xmax, ymax)
            crop_image = self.transform(crop_image)
            crop_image = crop_image.unsqueeze(0)
            imgs.append(crop_image)

        imgs = torch.cat(imgs, dim=0)
        features = self.extract_feature(imgs)

        labels = self.calculate(features, top_k=self._top_k)
        results = np.concatenate((bounding_box, labels), axis=1)
        return results

    def most_frequent(self, ids, scores):
        counter = 0
        num = ids[0]
        score = 1.0 - scores[0]
        for _, i in enumerate(ids):
            curr_frequency = ids.count(i)
            if (curr_frequency > counter):
                counter = curr_frequency
                num = i
                score = 1.0 - scores[_]
        return np.array([score, num])

    def calculate(self, features, metric='cosine', top_k=1):
        distance = np.power(cdist(features, self.gallery, metric=metric), 2).astype(np.float16)
        ids = np.argsort(distance).astype(np.int32)[:, :top_k]
        result = []
        for i, id in enumerate(ids):
            result.append(self.most_frequent(self.gallery_id[id].tolist(), distance[i][id].tolist()))
        return result

def is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def cv2_PIL(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


def crop(img, xmin, ymin, xmax, ymax):
    if not is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((xmin, ymin, xmax, ymax))

def load_gallery(gallery_path, gallery_ids_path):
    with open(gallery_path, 'rb') as f:
        gallery = pickle.load(f)
    with open(gallery_ids_path, 'rb') as f:
        gallery_id_path = pickle.load(f)
    return gallery, gallery_id_path

def load_checkpoint(fpath):
    if fpath is None:
        raise ValueError('File path is None')
    if not os.path.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint

def load_pretrained_weights(model, weight_path):
    checkpoint = load_checkpoint(weight_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:] # discard module.
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(weight_path)
        )
    else:
        print(
            'Successfully loaded pretrained weights from "{}"'.
            format(weight_path)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )
