####################################################################################################
# TANSmodels: Task-Adaptive Neural Network Search with Meta-Contrastive Learning
# Wonyong Jeong, Hayeon Lee, Geon Park, Eunyoung Hyung, Jinheon Baek, Sung Ju Hwang
# github: https://github.com/wyjeong/TANS, email: wyjeong@kaist.ac.kr
####################################################################################################
import os
import glob
import torch
import time
import random
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from misc.utils import *


def get_loader(args, mode='train'):
    if mode=='train':
        dataset = MetaTrainDataset(args, mode='train')
        loader = DataLoader(dataset=dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=0)
    else:
        dataset = MetaTrainDataset(args, mode='test')
        loader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=0)
    return dataset, loader


def get_meta_test_loader(args, mode='train'):
    dataset = MetaTestDataset(args, mode=mode)
    loader = DataLoader(dataset=dataset,
                        batch_size=args.batch_size,
                        shuffle=(mode == 'train'),
                        num_workers=0)
    return dataset, loader


class MetaTestDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        self.datapath=self.args.testdatapath
        self.dataset_list = os.listdir(self.datapath)
        self.data = torch_load(os.path.join(self.datapath, self.dataset_list[0]))
        self.query=self.loaddata()
        self.curr_dataset = self.dataset_list[0]

    def set_mode(self, mode):
        self.mode = mode

    def get_dataset_list(self):
        return self.dataset_list

    def set_dataset(self, dataset):
        self.curr_dataset = dataset
        self.data = torch_load(os.path.join(self.datapath, dataset+'.pth'))
        print(f"{dataset}:  #_test: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = index%4
        return x, y

    def get_query_set(self, task):
        return self.query[task+'.pth']

    def get_n_clss(self):
        return self.data['nclss']
    def loaddata(self):
        imgss = {}
        for dataset in self.dataset_list:
            dataemb = torch_load(os.path.join(self.datapath, dataset))
            imgs = torch.stack(dataemb).reshape(-1,1000)
            imgss[dataset] = imgs
        return imgss


class MetaTrainDataset(Dataset):

    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        self.model_zoo_path = self.args.model_zoo
        self.model_list = os.listdir(self.model_zoo_path)
        self.data_path = self.args.traindatapath
        self.test_path=self.args.testdatapath
        if self.mode=='train':
            self.query = self.loaddata()
            self.dataset_list = os.listdir(self.data_path)
        else :
            self.query= self.loadtestdata()
            self.dataset_list = os.listdir(self.test_path)

        start_time = time.time()
        self.contents = []
        self.loadmod()
        self.ind=0
        print(f"{len(self.contents)} pairs loaded ({time.time() - start_time:.3f}s) ")

    def loadmod(self):
        for dataset in self.model_list:
            modelemb = torch_load(os.path.join(self.args.model_zoo, dataset))
            models = []
            models.append({
                'acc': 0.99,
                'f_emb': modelemb.squeeze(1),
            })
            self.contents.append((dataset[:-4], models))

    # def loaddata(self):
    #     trainimgss = {}
    #     testimgss = {}
    #     for dataset in self.model_list:
    #         imgs= torch_load(os.path.join(self.data_path, dataset[:-4] + '.pt'))
    #         trainimgss[dataset[:-4]] = imgs[:-40]
    #         testimgss[dataset[:-4]] = imgs[-40:]
    #
    #     return trainimgss, testimgss

    def loaddata(self):
        trainimgss = {}
        for dataset in self.model_list:
            data = torch_load(os.path.join(self.data_path, dataset[:-4] + '.pt'))
            keyss = list(data.keys())
            cls1, cls2, cls3, cls4 = data[keyss[0]], data[keyss[1]], data[keyss[2]], data[keyss[3]]
            imgs = torch.cat([cls1.unsqueeze(0), cls2.unsqueeze(0), cls3.unsqueeze(0), cls4.unsqueeze(0)], 0)
            imgs = imgs.transpose(0, 1)
            imgs=imgs.reshape(-1,3,64,64)
            trainimgss[dataset[:-4]] = imgs
            # testimgss[dataset[:-4]] = imgs[-120:]
        return trainimgss

    def loadtestdata(self):
        testimgss = {}
        # testimgss={}
        for dataset in self.model_list:
            data = torch_load(os.path.join(self.test_path, dataset[:-4] + '.pt'))
            keyss = list(data.keys())
            cls1, cls2, cls3, cls4 = data[keyss[0]], data[keyss[1]], data[keyss[2]], data[keyss[3]]
            imgs = torch.cat([cls1.unsqueeze(0), cls2.unsqueeze(0), cls3.unsqueeze(0), cls4.unsqueeze(0)], 0)
            imgs = imgs.transpose(0, 1)
            imgs=imgs.reshape(-1,3,64,64)
            testimgss[dataset[:-4]] = imgs
            # testimgss[dataset[:-4]] = imgs[-120:]
        return testimgss

    def __len__(self):
        return len(self.contents)

    def set_mode(self, mode):
        self.mode = mode

    def __getitem__(self, index):
        dataset = self.contents[index][0]
        n_models = len(self.contents[index][1])
        if n_models == 1:
            idx = 0
        else:
            idx = random.randint(0, n_models - 1)
        model = self.contents[index][1][idx]
        acc = model['acc']
        f_emb = model['f_emb'].squeeze(1)
        return dataset, acc, f_emb

    def get_query(self, datasets):
        x_batch = []
        for d in datasets:
            totlen=len(self.query[d])
            if self.ind + 21 > totlen:
                self.ind = 0
            x = self.query[d][self.ind%totlen:(self.ind+10)%totlen]
            x_batch.append(x)
        self.ind=self.ind+10
        return x_batch
    def get_modellist(self):
        return self.model_list




