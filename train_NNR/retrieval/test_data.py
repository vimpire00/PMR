####################################################################################################
# TANSmodels: Task-Adaptive Neural Network Search with Meta-Contrastive Learning
# Wonyong Jeong, Hayeon Lee, Geon Park, Eunyoung Hyung, Jinheon Baek, Sung Ju Hwang
# github: https://github.com/wyjeong/TANS, email: wyjeong@kaist.ac.kr
####################################################################################################

import os
import sys
import glob
import time
import atexit
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from retrieval.loss import HardNegativeContrastiveLoss
from retrieval.measure import compute_recall
import torch.nn as nn
import torchvision
from misc.utils import *
from data.loader import *
from retrieval.nets import *
import numpy as np
# from retrieval.meature import compute_recall
def generate_JLmatrix(d,k):
    a = torch.randn([k, d], out=None)
    b = torch.div(torch.tensor(1), torch.sqrt(torch.tensor(k))) * a
    return b

def JLTransform(test,R):
    transformed_matrix = torch.matmul(R, test.t())
    return transformed_matrix

def compute_recall(query_embs, model_embs, npts=None):
    if npts is None:
        npts = query_embs.shape[0]
    index_list = []
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        # Get query_embs image
        query_emb = query_embs[index].reshape(1, query_embs.shape[1])
        # Compute scores
        d = np.dot(query_emb, model_embs.squeeze(1).T).flatten()
        sorted_index_lst = np.argsort(d)[::-1]
        index_list.append(sorted_index_lst[0])
        # Score
        rank = np.where(sorted_index_lst == index)[0][0]
        ranks[index] = rank
        top1[index] = sorted_index_lst[0]
    recalls = {}
    for v in [1, 5, 10, 50, 100]:
        recalls[v] = 100.0 * len(np.where(ranks < v)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return recalls, medr, meanr

def compute_recall_JL(out_indice):
        # Get query_embs image
        # Compute scores
    index_list = []
    npts=out_indice.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        sorted_index_lst=out_indice[index]
        index_list.append(sorted_index_lst[0])
        # Score
        rank = np.where(sorted_index_lst == index)[0][0]
        ranks[index] = rank
        top1[index] = sorted_index_lst[0]
    recalls = {}
    for v in [1, 5, 10, 50, 100]:
        recalls[v] = 100.0 * len(np.where(ranks < v)[0]) / len(ranks)
    return recalls

class Test:

    def __init__(self, args):
        self.args = args
        self.parameters = []
        self.device = args.device
        self.resnet = torchvision.models.resnet18(pretrained=True).to(self.device).eval()
        atexit.register(self.atexit)

    def atexit(self):
        print('Process destroyed.')

    def test(self):
        print(f'Begin test process')
        start_time = time.time()
        self.init_loaders_for_meta_test()
        self.load_encoders_for_meta_test()
        # self.load_cross_modal_space()
        # self.meta_test()
        self.evaluate()
        # self.eva_JL()
        print(f'Process done ({time.time() - start_time:.2f})')

    def init_loaders_for_meta_test(self):
        print('==> loading meta-test loaders')
        # self.tr_dataset, self.tr_loader = get_meta_test_loader(self.args, mode='train')
        self.te_dataset, self.te_loader = get_loader(self.args, mode='test')

    def load_encoders_for_meta_test(self):
        print('==> loading encoders ... ')
        # _loaded = torch_load(os.path.join(self.args.load_path,'./szysaved_model_ba15_ep5.pt'))
        # _loaded = torch_load(os.path.join(self.args.load_path,'szy3saved_026.pt'))
        _loaded = torch_load(os.path.join(self.args.load_path,self.args.load_model))
        # _loaded = torch_load(os.path.join(self.args.load_path,'./szysaved_model_max_recall_ba58_ep5.pt'))
        self.enc_q = QueryEncoder(self.args).to(self.device).eval()
        self.enc_q.load_state_dict(_loaded['enc_q'])
        self.enc_m = ModelEncoder(self.args).to(self.device).eval()
        self.enc_m.load_state_dict(_loaded['enc_m'])
        # self.predictor = PerformancePredictor(self.args).to(self.device).eval()
        # self.predictor.load_state_dict(_loaded['predictor'])

    def load_cross_modal_space(self):
        print('==> loading the cross modal space ... ')
        self.cross_modal_info = torch_load(os.path.join(self.args.load_path,'retrieval.pt'))
        self.datasetlist=self.cross_modal_info['dataset']
        # self.m_embs=m_embs

    def meta_test(self):
        print('==> meta-testing on unseen datasets ... ')
        query_id=0
        query_list=self.datasetlist
        for  query_dataset in query_list:
            query_id=query_id+1
            self.te_dataset.set_dataset(query_dataset)
            self.query_dataset = query_dataset
            self.meta_test_results = {
                'query': self.query_dataset,
                'retrieval': {},
            }
            query = self.te_dataset.get_query_set(self.query_dataset)
            query = torch.stack([d.to(self.device) for d in query])
            q_emb = self.get_query_embeddings(query).unsqueeze(0)
            print(f' [query_id:{query_id }] query by {query_dataset} ... ')
            print(f' ========================================================================================================================')
            retrieved = self.retrieve(q_emb, self.args.n_retrievals)

    def retrieve(self, _q, n_retrieval):
        s_t = time.time()
        m_embs = torch.stack(self.cross_modal_info['m_emb']).to(self.device).squeeze()
        scores = torch.mm(_q, m_embs.squeeze().t()).squeeze()
        sorted, sorted_idx = torch.sort(scores, 0, descending=True)

        top_10_idx = sorted_idx[:n_retrieval]
        retrieved = {}
        for idx in top_10_idx:
            for ii in range(m_embs.shape[0]):
                dataset=self.cross_modal_info['dataset'][ii]
                m_emb=self.cross_modal_info['m_emb'][ii]
                if torch.equal(self.cross_modal_info['m_emb'][idx],m_emb):
                    print(dataset,"OK")
                    break
        return retrieved

    def save_meta_test_results(self, query_dataset):
        f_write(self.args.log_path, f'meta_test{query_dataset}.txt', {
            'options': vars(self.args),
            'results': self.meta_test_results
        })

    def get_query_embeddings(self, x_emb):
        print(' ==> converting dataset to query embedding ... ')
        q = self.enc_q(x_emb.unsqueeze(0))
        return q.squeeze()

    def   EuclideanDistance(self,t1, t2):
        dim = len(t1.size())
        if dim == 2:
            N, C = t1.size()
            M, _ = t2.size()
            dist = -2 * torch.matmul(t1, t2.permute(1, 0))
            dist += torch.sum(t1 ** 2, -1).view(N, 1)
            dist += torch.sum(t2 ** 2, -1).view(1, M)
            dist = torch.sqrt(dist)
            return dist
        elif dim == 3:
            B, N, _ = t1.size()
            _, M, _ = t2.size()
            dist = -2 * torch.matmul(t1, t2.permute(0, 2, 1))
            dist += torch.sum(t1 ** 2, -1).view(B, N, 1)
            dist += torch.sum(t2 ** 2, -1).view(B, 1, M)
            dist = torch.sqrt(dist)
            return dist
        else:
            print('error...')


    def evaluate(self):
        dataset, acc, f_emb = next(iter(self.te_loader))
        with torch.no_grad():
            query = []
            queryimgs = self.te_dataset.get_query(dataset)
            for queryimg in queryimgs:
                queryimg = queryimg.to(self.device)
                queryemb = self.resnet(queryimg)
                query.append(queryemb)
            query = [d.to(self.device) for d in query]
            q_emb = self.enc_q(query)
            # print(query[0].shape,q_emb.shape,"shape111")
            m_emb = self.enc_m(f_emb.to(self.device))
            # q, m, lss = self.forward(f_emb, query)
        print("plain")
        R, medr, meanr = compute_recall(q_emb.cpu(), m_emb.cpu())
        embs={}
        emb_length=m_emb.shape[-1]
        for k in range(len(dataset)):
            embs[dataset[k]] = m_emb[k]
        print("savingmodelhash")
        torch.save(embs, './tansembs_da618' + str(emb_length) + '.pt')
        print(
              f'R@1 {R[1]:.1f} , R@5 {R[5]:.1f}, R@10 {R[10]:.1f}, R@50 {R[50]:.1f} ' )

    def eva_JL(self):
        dataset, acc, f_emb = next(iter(self.te_loader))
        n=[58.0]
        d=512
        eps=0.1
        k=torch.div(1,eps)*torch.div(1,eps)*torch.log(torch.Tensor(n))
        k=int(k)
        print(k,"kk")
        JLmatrix = generate_JLmatrix(d,k)
        # pdist = nn.PairwiseDistance(p=2)  # p=2就是计算欧氏距离，p=1就是曼哈顿距离，例如上面的例子，距离是1.
        # transformed_data = JLTransform(data, JLmatrix)
        # transformed_query = JLTransform(query, JLmatrix)
        with torch.no_grad():
            query = []
            # query=self.te_dataset.get_query(dataset)
            queryimgs = self.te_dataset.get_query(dataset)
            for queryimg in queryimgs:
                queryimg = queryimg.to(self.device)
                queryemb = self.resnet(queryimg)
                query.append(queryemb)
            query = [d.to(self.device) for d in query]
            q_emb = self.enc_q(query)
            m_emb=f_emb.squeeze().to(self.device)
            # m_emb = self.enc_m(f_emb.to(self.device))
            dist1 = self.EuclideanDistance(q_emb.squeeze(), m_emb.squeeze())
            out_sorted, out_indices = torch.sort(dist1)
            R1=compute_recall_JL(out_indices)
            print(
                f'plR@1 {R1[1]:.1f} , R@5 {R1[5]:.1f}, R@10 {R1[10]:.1f}, R@50 {R1[50]:.1f} ')
            q_emb=q_emb
            m_emb=m_emb
            t_q=JLTransform(q_emb, JLmatrix)
            t_m=JLTransform(m_emb.squeeze(), JLmatrix)
            dist3 = self.EuclideanDistance(t_q.t(),t_m.t())
            out_sorted, out_indices = torch.sort(dist3)
        R=compute_recall_JL(out_indices)
        print(
            f'JLR@1 {R[1]:.1f} , R@5 {R[5]:.1f}, R@10 {R[10]:.1f}, R@50 {R[50]:.1f} ')

    def eva_JL2(self):
        dataset, acc, f_emb = next(iter(self.te_loader))
        with torch.no_grad():
            query = []
            # query=self.te_dataset.get_query(dataset)
            queryimgs = self.te_dataset.get_query(dataset)
            for queryimg in queryimgs:
                queryimg = queryimg.to(self.device)
                queryemb = self.resnet(queryimg)
                query.append(queryemb)
            query = [d.to(self.device) for d in query]
            q_emb = self.enc_q(query)
            m_emb = self.enc_m(f_emb.to(self.device))
            q_jl = jl_transform(q_emb, 800, "basic")
            m_jl = jl_transform(m_emb, 800, "basic")
            tr_basic_dist = distance_dataset(index, jlt_basic_transformed)




