from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os.path as osp
import csv

from .bases import BaseImageDataset


class SATUDORA(BaseImageDataset):

    dataset_dir = 'satudora'

    def __init__(self, root='data', verbose=True, **kwargs):
        super(SATUDORA, self).__init__(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        # self.train_dir = osp.join(self.dataset_dir, 'MSMT17_V1/train')
        # self.test_dir = osp.join(self.dataset_dir, 'MSMT17_V1/test')
        self.csv_train_path = osp.join(self.dataset_dir, 'train.csv')
        self.csv_query_path = osp.join(self.dataset_dir, 'test.csv')
        self.csv_gallery_path = osp.join(self.dataset_dir, 'train.csv')

        self._check_before_run()
        # train = self._process_dir(self.train_dir, self.list_train_path)
        # #val = self._process_dir(self.train_dir, self.list_val_path)
        # query = self._process_dir(self.test_dir, self.list_query_path)
        # gallery = self._process_dir(self.test_dir, self.list_gallery_path)

        train = self._create_dataset(self.csv_train_path,0)
        query = self._create_dataset(self.csv_query_path,1)
        gallery = self._create_dataset(self.csv_gallery_path,2)


        if verbose:
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.csv_train_path):
            raise RuntimeError("'{}' is not available".format(self.csv_train_path))
        if not osp.exists(self.csv_query_path):
            raise RuntimeError("'{}' is not available".format(self.csv_query_path))


    def _create_dataset(self, file_path,camid = 0):
        dataset = []
        pid_container = set()
        with open(file_path, 'rt') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                img_path, pid = row[0].split(',')
                pid = int(pid)
                dataset.append((img_path, pid, camid))
                pid_container.add(pid)
            num_pids = len(pid_container)
            #     # check if pid starts from 0 and increments with 1
            # for idx, pid in enumerate(pid_container):
            #     assert idx == pid, "See code comment for explanation"
        return dataset
