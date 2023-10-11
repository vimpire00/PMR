####################################################################################################
#Privacy-Preserving and Accurate Deep Neural Network Retrieval
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


class TestS:

    def __init__(self, args):
        self.args = args
        self.parameters = []
        self.device = 'cpu'
        self.hash_vals,self.hash_name=self.load_modelhash()
        self.dataset=self.hash_name

    def load_modelhash(self):
        hash_dict={}
        hashembs=torch.load(os.path.join(self.args.hash_path,self.args.hash_name))
        hash_name=list(hashembs.keys())
        hash_vals=list(hashembs.values())
        # print(hash_vals[0].shape,"shape")
        # hash_vals=hashembs['value']
        # hash_name=hashembs['name']
        # hash_name=list(hashembs['name'])
        # modelembs=list(hashembs['value'])
        # for i in range(len(hash_name)):
        #     hash_dict[hash_name[i]]=hash_vals[i]
        return hash_vals,hash_name

    def eval_plain(self,queryembss):
        hash_embs=torch.stack(self.hash_vals)
        m_emb=hash_embs
        querytname=list(self.dataset)
        modelname=self.hash_name
        r=0
        query_embs=queryembss
        q_emb= query_embs
        d = np.dot(q_emb.squeeze(), m_emb.T)
        for j in range(d.shape[0]):
            d_j=d[j]
            sorted_index_lst = np.argsort(d_j)[::-1]
            if modelname[sorted_index_lst[0]]==querytname[j]:
                r=r+1
        print("score",r/58.0)
