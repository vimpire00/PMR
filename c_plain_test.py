####################################################################################################
#Improved PMR
####################################################################################################
import logging
import os
import sys
import glob
import time
import atexit
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from loss import HardNegativeContrastiveLoss
from measure import compute_recall
import torchvision
from utils_tans import *
from data_loader import *
from retrieval_nets import *
import numpy as np
from operator import itemgetter
from mpc_tans import run_experiment

class TestC:

    def __init__(self, args):
        self.args = args
        self.parameters = []
        self.device = 'cpu'
        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.resnet.load_state_dict(torch.load('./res18.pth'))
        self.resnet.to(self.device).eval()
        self.test()

    def test(self):
        print(f'init_loader,init_query_encoder')
        start_time = time.time()
        self.init_loaders_for_meta_test()
        self.load_encoders_for_query_encoder()

    def init_loaders_for_meta_test(self):
        print('==> loading meta-test loaders')
        self.te_dataset, self.te_loader = get_loader(self.args, mode='test')

    def load_encoders_for_query_encoder(self):
        print('==> loading query encoders ... ')
        _loaded = torch_load(os.path.join(self.args.load_path, self.args.model))
        self.enc_q = QueryEncoder(self.args).to(self.device)
        self.enc_q.load_state_dict(_loaded['enc_q'])
        self.enc_q.eval()
    
    def load_query_data(self):

        q_embs=[]
        q_imgs=[]
        dataset, _, _ = next(iter(self.te_loader))
        self.dataset=dataset
        with torch.no_grad():
            queryimgs = self.te_dataset.get_query(dataset)
            for queryimg in queryimgs:
                query=[]
                queryimg = queryimg.to(self.device)
                queryemb = self.resnet(queryimg)
                query.append(queryemb)
                q_imgs.append(queryimg)
                query = [d.to(self.device) for d in query]
                q_emb = self.enc_q(query)
                q_embs.append(q_emb)

        return q_embs,dataset,q_imgs