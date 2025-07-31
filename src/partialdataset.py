# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import random
from logging import getLogger

import cv2
from PIL import ImageFilter, Image
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import scipy.io as sio
import os
import json
from utils.config import args
from numpy.testing import assert_array_almost_equal
import h5py
logger = getLogger()

class cross_modal_dataset(data.Dataset):
    def __init__(self, dataset, partial_rate,  mode, partial_file=None, partial_mode='sym', root_dir='./data/', pred=False, probability=[], log=''):
        self.r = partial_rate # noise ratio
        self.mode = mode
        doc2vec = True
        if 'wiki' in dataset.lower():
            root_dir = os.path.join(root_dir, 'wiki')
            path = os.path.join(root_dir, 'wiki_deep_doc2vec_data_corr_ae.h5py')
            valid_len = 231
        elif 'nus' in dataset.lower():
            root_dir = os.path.join(root_dir, 'nus')
            path = os.path.join(root_dir, 'nus_wide_deep_doc2vec_data_42941.h5py')
            valid_len = 5000
        elif 'inria' in dataset.lower():
            root_dir = os.path.join(root_dir, 'inria')
            path = os.path.join(root_dir, 'INRIA-Websearch.mat')
            doc2vec = False
        elif 'xmedianet4view' in dataset.lower():
            root_dir = os.path.join(root_dir, 'XMediaNet4View')
            path = os.path.join(root_dir, 'XMediaNet4View_pairs.mat')
            doc2vec = False
        elif 'xmedianet2views' in dataset.lower():
            root_dir = os.path.join(root_dir, 'xmedianet2views')
            path = os.path.join(root_dir, 'xmedianet_deep_doc2vec_data.h5py')
            valid_len = 4000
        else:
            raise Exception('Have no such dataset!')

        if doc2vec:
            h = h5py.File(path)
            if self.mode == 'test' or self.mode == 'valid':
                test_imgs_deep = h['test_imgs_deep'][()].astype('float32')
                test_imgs_labels = h['test_imgs_labels'][()]
                test_imgs_labels -= np.min(test_imgs_labels)
                try:
                    test_texts_idx = h['test_text'][()].astype('float32')
                except Exception as e:
                    test_texts_idx = h['test_texts'][()].astype('float32')
                test_texts_labels = h['test_texts_labels'][()]
                test_texts_labels -= np.min(test_texts_labels)
                test_data = [test_imgs_deep, test_texts_idx]
                test_labels = [test_imgs_labels, test_texts_labels]

                valid_flag = True
                try:
                    valid_texts_idx = h['valid_text'][()].astype('float32')
                except Exception as e:
                    try:
                        valid_texts_idx = h['valid_texts'][()].astype('float32')
                    except Exception as e:
                        valid_flag = False
                        valid_data = [test_data[0][0: valid_len], test_data[1][0: valid_len]]
                        valid_labels = [test_labels[0][0: valid_len], test_labels[1][0: valid_len]]

                        test_data = [test_data[0][valid_len::], test_data[1][valid_len::]]
                        test_labels = [test_labels[0][valid_len::], test_labels[1][valid_len::]]
                if valid_flag:
                    valid_imgs_deep = h['valid_imgs_deep'][()].astype('float32')
                    valid_imgs_labels = h['valid_imgs_labels'][()]
                    valid_texts_labels = h['valid_texts_labels'][()]
                    valid_texts_labels -= np.min(valid_texts_labels)
                    valid_data = [valid_imgs_deep, valid_texts_idx]
                    valid_labels = [valid_imgs_labels, valid_texts_labels]

                train_data = valid_data if self.mode == 'valid' else test_data
                train_label = valid_labels if self.mode == 'valid' else test_labels
            elif self.mode == 'train':
                tr_img = h['train_imgs_deep'][()].astype('float32')
                tr_img_lab = h['train_imgs_labels'][()]
                tr_img_lab -= np.min(tr_img_lab)
                try:
                    tr_txt = h['train_text'][()].astype('float32')
                except Exception as e:
                    tr_txt = h['train_texts'][()].astype('float32')
                tr_txt_lab = h['train_texts_labels'][()]
                tr_txt_lab -= np.min(tr_txt_lab)
                train_data = [tr_img, tr_txt]
                train_label = [tr_img_lab, tr_txt_lab]
            else:
                raise Exception('Have no such set mode!')
            h.close()


        else:
            data = sio.loadmat(path)
            if 'xmedianet4view' in dataset.lower():
                if self.mode == 'train':
                    train_data = [data['train'][0, v].astype('float32') for v in range(4)]
                    train_label = [data['train_labels'][0, v].reshape([-1]).astype('int64') for v in range(4)]
                elif self.mode == 'valid':
                    train_data = [data['valid'][0, v].astype('float32') for v in range(4)]
                    train_label = [data['valid_labels'][0, v].reshape([-1]).astype('int64') for v in range(4)]
                elif self.mode == 'test':
                    train_data = [data['test'][0, v].astype('float32') for v in range(4)]
                    train_label = [data['test_labels'][0, v].reshape([-1]).astype('int64') for v in range(4)]
                else:
                    raise Exception('Have no such set mode!')
            else:
                if self.mode == 'train':
                    train_data = [data['tr_img'].astype('float32'), data['tr_txt'].astype('float32')]
                    train_label = [data['tr_img_lab'].reshape([-1]).astype('int64'), data['tr_txt_lab'].reshape([-1]).astype('int64')]
                elif self.mode == 'valid':
                    train_data = [data['val_img'].astype('float32'), data['val_txt'].astype('float32')]
                    train_label = [data['val_img_lab'].reshape([-1]).astype('int64'), data['val_txt_lab'].reshape([-1]).astype('int64')]
                elif self.mode == 'test':
                    train_data = [data['te_img'].astype('float32'), data['te_txt'].astype('float32')]
                    train_label = [data['te_img_lab'].reshape([-1]).astype('int64'), data['te_txt_lab'].reshape([-1]).astype('int64')]

                else:
                    raise Exception('Have no such set mode!')



        train_label = [la.astype('int64') for la in train_label]
        noise_label = train_label

        if partial_file is None:
            if partial_mode == 'sym':
                partial_file = os.path.join(root_dir, 'partial_labels_%g_sym.json' % self.r)
            elif partial_mode == 'asym':
                partial_file = os.path.join(root_dir, 'partial_labels_%g__asym.json' % self.r)


        partialY = [[],[]]

        if self.mode == 'train':
            if os.path.exists(partial_file):
                partialY = json.load(open(partial_file, "r"))
                self.class_num = np.unique(noise_label).shape[0]
            else:
                partialY[0] = generate_uniform_cv_candidate_labels(torch.tensor(train_label[0]), partial_rate).tolist()
                partialY[1] = generate_uniform_cv_candidate_labels(torch.tensor(train_label[1]), partial_rate).tolist()
                self.class_num = np.unique(noise_label).shape[0]

                json.dump(partialY, open(partial_file, "w"))

        self.train_label = np.array(train_label)
        self.default_train_data = train_data
        self.train_data = self.default_train_data

        if self.mode == 'train':
            self.default_partial_label = np.array(partialY)
            self.partial_label = self.default_partial_label
        else:
            self.partial_label = self.train_label

        if pred:
            self.prob = [np.ones_like(ll) for ll in self.default_partial_label]
        else:
            self.prob = None


    def __getitem__(self, index):

        if self.prob is None:
            return [self.train_data[v][index] for v in range(len(self.train_data))], [self.partial_label[v][index] for v in range(len(self.train_data))], [self.train_label[v][index] for v in range(len(self.train_data))], index

        else:
            return [self.train_data[v][index] for v in range(len(self.train_data))], [self.partial_label[v][index] for v in range(len(self.train_data))], [self.train_label[v][index] for v in range(len(self.train_data))], [self.prob[v][index] for v in range(len(self.prob))], index

    def __len__(self):
        return len(self.train_data[0])

def generate_uniform_cv_candidate_labels(train_labels, partial_rate=0.1):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    transition_matrix =  np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))] = partial_rate

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):
        partialY[j, :] = torch.from_numpy((random_n[j, :] < transition_matrix[train_labels[j], :]) * 1)

    print(partialY)
    print("Finish Generating Candidate Label Sets!\n")
    return partialY

