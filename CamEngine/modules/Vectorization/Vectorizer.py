from __future__ import absolute_import
from .torchreid.utils import FeatureExtractor

class Vectorizer(object):

    def __init__(self, model_path, model_name, device='cpu'):
        self._model_path = model_path
        self._feature_extractor = FeatureExtractor(model_name=model_name, model_path=model_path, device=device)

    def predict(self, images):
        return self._feature_extractor(images)

if __name__ == '__main__':
    vectorizr = Vectorizer("./models/osnet_ain_x1_0_msmt17.pth", "osnet_ain_x1_0", "cuda")
