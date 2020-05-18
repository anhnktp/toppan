from __future__ import print_function
from __future__ import division

import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.transforms import *
from .Base import RecognitionBase
from .args import argument_parser, image_dataset_kwargs
from .torchreid.data_manager import ImageDataManager
from .torchreid import models
from .torchreid.utils.iotools import check_isfile
from .torchreid.utils.torchtools import load_pretrained_weights
from .recognition_config import *
from ..Common.utils import cv2_PIL, crop
import pickle

parser = argument_parser()
args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = "2"

class Recognition(RecognitionBase):

    def __init__(self, device):
        super(Recognition, self).__init__()

        self._device = device


        cudnn.benchmark = True

        dm = ImageDataManager(True, **image_dataset_kwargs(args))
        _, testloader_dict = dm.return_dataloaders()

        self.model = models.init_model(name=args.arch, num_classes=dm.num_train_pids, loss={'xent', 'htri'}).cuda(self._device)
        # Load model here
        self.model_file = os.path.join(os.path.dirname(__file__), 'model', MODEL_FILE)

        if check_isfile(self.model_file):
            load_pretrained_weights(self.model, self.model_file)

        dataset_name = "satudora"
        self.galleryloader = testloader_dict[dataset_name]['gallery']
        # Extract feature for training (gallery) set
        self.model.eval()
        self.gf, self.g_pids = self.extract_feature_train_set()

    def extract_feature_test_image(self, crop_images):
        with torch.no_grad():
            imagenet_mean = [0.485, 0.456, 0.406]
            imagenet_std = [0.229, 0.224, 0.225]
            normalize = Normalize(mean=imagenet_mean, std=imagenet_std)

            transform_test = Compose([
                Resize((TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGH)),
                ToTensor(),
                normalize,
            ])
            imgs = []
            for img in crop_images:
                img_tensor = transform_test(img)
                img_tensor = img_tensor.unsqueeze(0)
                imgs.append(img_tensor)
            qf, q_pids = [], []
            imgs = torch.cat(imgs, dim=0).cuda(self._device)
            features = self.model(imgs)
            features = features.data.cpu()
            qf.append(features)
            qf = torch.cat(qf, 0)
            return qf

    def extract_feature_train_set(self):
        gf, g_pids = [], []
        current_path = os.path.dirname(os.path.abspath(__file__))
        gf_file_path = os.path.join(current_path, "gf.pickle")
        pids_file_path = os.path.join(current_path, "g_pids.pickle")
        if os.path.exists(gf_file_path) and os.path.exists(pids_file_path):
            try:
                pickle_gf = open(gf_file_path, "rb")
                gf = pickle.load(pickle_gf)

                pickle_g_pids = open(pids_file_path, "rb")
                g_pids = pickle.load(pickle_g_pids)

            except FileNotFoundError:
                print("The pickle file does not exist")

        if len(g_pids) != 0:
            return gf, g_pids

        with torch.no_grad():
            gf, g_pids = [], []
            for batch_idx, (imgs, pids, camids, _) in enumerate(self.galleryloader):
                imgs = imgs.cuda(self._device)
                features = self.model(imgs)
                features = features.data.cpu()
                gf.append(features)
                g_pids.extend(pids)
            gf = torch.cat(gf, 0)
            g_pids = np.asarray(g_pids)
            pickle_gf = open(gf_file_path, "wb")
            pickle.dump(gf, pickle_gf)
            pickle_gpids = open(pids_file_path, "wb")
            pickle.dump(g_pids, pickle_gpids)
            pickle_gf.close()
            pickle_gpids.close()
        return gf, g_pids

    def get_multiple_labels_from_boxes(self, crop_images, top_k=5):
        # Extract feature for test (query) image
        self.qf = self.extract_feature_test_image(crop_images)
        m, n = self.qf.size(0), self.gf.size(0)
        distmat = torch.pow(self.qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(self.gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, self.qf, self.gf.t())
        distmat = distmat.numpy()
        indices = np.argsort(distmat, axis=1)
        labels = []
        for i in range(len(crop_images)):
            # labels.append(indices[i][:top_k])
            top_k_index = indices[i][:top_k]
            res = []
            for i in top_k_index:
                res.append(int(self.g_pids[i]))
            labels.append(res)
        return labels

    def getOutput(self, img, bounding_box):
        img = cv2_PIL(img)
        crop_images = []
        for box in bounding_box:
            xmin, ymin, xmax, ymax, score,  = box
            crop_image = crop(img, xmin, ymin, xmax, ymax)
            crop_images.append(crop_image)
        labels = self.get_multiple_labels_from_boxes(crop_images)
        return labels
        # for i, box in enumerate(bounding_box['boxes']):
        #     box['recognition'] = labels[i]
        return bounding_box
