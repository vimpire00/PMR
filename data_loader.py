
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
from utils_tans import *

def get_loader(args, mode='train'):
    # if mode=='train':
    #     dataset = MetaTrainDataset(args, mode=mode)
    #     loader = DataLoader(dataset=dataset,
    #                     batch_size=args.batch_size,
    #                     shuffle=True,
    #                     num_workers=0)
    # else:
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

class MetaTrainDataset(Dataset):

    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        self.model_list=os.listdir(self.args.model_zoo)
        self.test_path=self.args.testdatapath
        # if self.mode=='train':
        #     self.query = self.loaddata()
        #     self.dataset_list = os.listdir(self.data_path)
        # else :
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

    def loadtestdata(self):
        testimgss = {}
        for dataset in self.model_list:
            data = torch_load(os.path.join(self.test_path, dataset[:-4] + '.pt'))
            keyss = list(data.keys())
            cls1, cls2, cls3, cls4 = data[keyss[0]], data[keyss[1]], data[keyss[2]], data[keyss[3]]
            imgs = torch.cat([cls1.unsqueeze(0), cls2.unsqueeze(0), cls3.unsqueeze(0), cls4.unsqueeze(0)], 0)
            imgs = imgs.transpose(0, 1)
            imgs=imgs.reshape(-1,3,64,64)
            testimgss[dataset[:-4]] = imgs

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
            x = self.query[d][self.ind%totlen:(self.ind+20)%totlen]
            x_batch.append(x)
        self.ind=self.ind+10
        return x_batch

    def get_modellist(self):
        return self.model_list


class TrainData(Dataset):
    def __init__(self, datapath):
        super(TrainData, self).__init__()
        self.resize = 64
        self.path = datapath
        self.data = torch.load(self.path)
        self.keys=list(self.data.keys())
        self.images, self.labels = self.loaddata(self.data)
    def __len__(self):
        return len(self.labels)

    def loaddata(self,data):
        cls1,cls2,cls3,cls4=data[self.keys[0]],data[self.keys[1]],data[self.keys[2]],data[self.keys[3]]
        len=cls1.shape[0]
        labels=torch.tensor([0,1,2,3])
        imgs=torch.cat([cls1.unsqueeze(0),cls2.unsqueeze(0),cls3.unsqueeze(0),cls4.unsqueeze(0)],0)
        imgs=imgs.transpose(0,1).reshape(-1,cls1.shape[-3],cls1.shape[-2],cls1.shape[-1])
        labels=labels.repeat(len)
        return imgs, labels

    def __getitem__(self, index):
        image = self.images[index,:,:,:]
        label = self.labels[index]
        return image,label
