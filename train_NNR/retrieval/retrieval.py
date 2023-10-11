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
import sys
sys.path.append('./')
from retrieval.loss import *
from retrieval.measure import compute_recall
from misc.utils import *
from data.loader import *
from retrieval.nets import *
import numpy as np
from retrieval.example2 import JLCal

class Retrieval:

    def __init__(self, args):
        self.args = args
        self.parameters = []
        self.device=args.device
        atexit.register(self.atexit)

    def atexit(self):
        print('Process destroyed.')

    def train(self):
        print(f'Begin train process')
        start_time = time.time()
        self.init_models()
        self.init_loaders()
        self.train_cross_modal_space()
        self.save_cross_modal_space()
        print(f'Process done ({time.time() - start_time:.2f})')
        sys.exit()

    def init_loaders(self):
        print('==> loading data loaders ... ')
        if self.args.mode=='train':
            self.tr_dataset, self.tr_loader = get_loader(self.args, mode='train')
            self.te_dataset, self.te_loader = get_loader(self.args, mode='test')
        # if self.args.mode=='test':
        else:
            self.te_dataset, self.te_loader = get_loader(self.args, mode='test')

    def init_models(self):
        print('==> loading encoders ... ')
        self.enc_m = ModelEncoder(self.args).to(self.device)
        self.enc_q = QueryEncoder(self.args).to(self.device)
        self.parameters = [*self.enc_q.parameters(), *self.enc_m.parameters()]
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.args.lr)
        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.resnet.load_state_dict(torch.load('./../SZY7/res18.pth'))
        self.resnet.to(self.device).eval()
        self.noise=torch.ones([1,3,64,64]).to(self.device)
        self.resemb=self.resnet(self.noise)
        self.criterion1=MMDLoss()
        self.criterion = HardNegativeContrastiveLoss(nmax=1, contrast=True)
        # self.criterion_mse = torch.nn.MSELoss()
        # self.cossim=torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        if not self.args.load_model==None:
            print("test")
            _loaded = torch_load(os.path.join(self.args.load_path,self.args.load_model))
            self.enc_q.load_state_dict(_loaded['enc_q'])
            self.enc_m.load_state_dict(_loaded['enc_m'])
        if self.args.mode=='test' or self.args.mode=='test_end':
            self.enc_q.eval()
            self.enc_m.eval()


    def train_cross_modal_space(self):
        print('==> train the cross modal space from model-zoo ... ')
        self.scores = {
            'tr_lss': [], 'te_lss': [],
            'r_1': [], 'r_5': [], 'r_10': [], 'r_50': [],
            'mean': [], 'median': [], 'mse': []}
        max_recall = 0
        start_time = time.time()
        for curr_epoch in range(self.args.n_epochs):
            query=[]
            ep_time = time.time()
            self.curr_epoch = curr_epoch
            ##################################################
            dataset, acc, f_emb = next(iter(self.tr_loader))  # 1 iteration == 1 epoch
            f_emb=f_emb.squeeze()
            queryimgs = self.tr_dataset.get_query(dataset)
            for queryimg in queryimgs:
                queryimg=queryimg.to(self.device)
                queryemb=self.resnet(queryimg)
                query.append(queryemb)
            q, m, lss= self.forward(f_emb, query)
            self.optimizer.zero_grad()
            loss1 = lss
            loss1.backward(retain_graph=True)
            self.optimizer.step()
            ##################################################
            tr_lss = loss1.item()
            if (curr_epoch + 1) % 10 == 0:
                r_1=0
                for i in range(10):
                    te_lss, R, medr, meanr = self.evaluate()
                    r_1=r_1+R[1]
                r_1=r_1/10.0
                print("evaluate","r_1",r_1)
                print(f'ep:{self.curr_epoch}, ' +
                      f'tr_lss: {tr_lss:.3f}, ' +
                      f'te_lss:{te_lss:.3f}, ' +
                      f'R@1 {R[1]:.1f} ({max_recall:.1f}), R@5 {R[5]:.1f}, R@10 {R[10]:.1f}, R@50 {R[50]:.1f} ' +
                      f'mean {meanr:.1f}, median {medr:.1f} ({time.time() - ep_time:.2f})')
                self.scores['tr_lss'].append(tr_lss)
                self.scores['te_lss'].append(te_lss)
                self.scores['r_1'].append(R[1])
                self.scores['r_5'].append(R[5])
                self.scores['r_10'].append(R[10])
                self.scores['r_50'].append(R[50])
                self.scores['median'].append(medr)
                self.scores['mean'].append(meanr)
                mse=0
                if r_1 > max_recall:
                    max_recall = r_1
                    self.save_model(True, curr_epoch, R, medr, meanr, mse)
                self.save_model(False, curr_epoch, R, medr, meanr, mse)
                print(f'==> training the cross modal space done. ({time.time() - start_time:.2f}s)')
                # self.evaluate2()

    def forward(self, f_emb, query):
        f_emb=f_emb.squeeze()
        resemb=self.resemb
        resemb=resemb.to(self.device)
        query = [d.to(self.device) for d in query]
        q_emb = self.enc_q(query)
        m_emb = self.enc_m(f_emb.to(self.device),resemb)
        loss1 = self.criterion1(q_emb, m_emb)
        loss2 = self.criterion(q_emb, m_emb)
        loss = loss2 + loss1
        return q_emb,m_emb,loss

    def l2norm(self, x):
        norm2 = torch.norm(x, 2, dim=1, keepdim=True)
        x = torch.div(x, norm2)
        return x

    def l1norm(self, x):
        norm1 = torch.norm(x, 1, dim=1, keepdim=True)
        x = torch.div(x, norm1)
        return x

    def l2norm2(self, x):
        norm2 = torch.norm(x, 2, dim=0, keepdim=True)
        x = torch.div(x, norm2)
        return x
    def evaluate2(self):
        # print("initmodels")
        self.init_loaders()
        self.init_models()
        r_1=0
        r_2=0
        with torch.no_grad():
            for i in range(10):
                dataset, acc, f_emb = next(iter(self.te_loader))
                f_emb = f_emb.squeeze()
                f_emb=f_emb.to(self.device)
                embs={}
                query = []
                queryimgs = self.te_dataset.get_query(dataset)
                for queryimg in queryimgs:
                    queryimg = queryimg.to(self.device)
                    queryemb = self.resnet(queryimg)
                    query.append(queryemb)
                q, m, lss1 = self.forward(f_emb, query)
                emb_length=m.shape[-1]
                if i == 0:
                    for k in range(len(dataset)):
                        embs[dataset[k]] = m[k]
                    torch.save(embs, './szymob13embs_ba10_da613'+str(emb_length)+'.pt')
                recalls, medr, meanr = compute_recall(q.cpu(), m.cpu())
                r_1=r_1+recalls[1]
            print(r_1 / 10.0, "r_1")

    def evaluate3(self):
        self.init_loaders()
        self.init_models()
        r_1 = 0
        r_2 = 0
        embs=torch.load(self.args.mobembpath,map_location=self.device)
        dataset=list(embs.keys())
        f_emb=torch.stack(list(embs.values()))
        with torch.no_grad():
            for i in range(10):
                f_emb = f_emb.squeeze()
                f_emb = f_emb.to(self.device)
                query = []
                queryimgs = self.te_dataset.get_query(dataset)
                for queryimg in queryimgs:
                    queryimg = queryimg.to(self.device)
                    queryemb = self.resnet(queryimg)
                    query.append(queryemb)
                q, m, lss1 = self.forward(f_emb, query)
                # q = self.l2norm2(q)
                # m = self.l2norm2(m)
                recalls, medr, meanr = compute_recall(q.cpu(), m.cpu())
                r_1 = r_1 + recalls[1]
                q=torch.stack(query).squeeze()
                JL = JLCal(1000, q.shape[-1], 128, "discrete")
                jlr = JL.comjlrecall(m.cpu(), q.cpu())
                r_2 = jlr + r_2
                # print(recalls, "recalls")
            print(r_1 / 5.0, "r_JL", r_2 / 5.0, "r_2")


    def evaluate(self):
        dataset, acc, f_emb = next(iter(self.te_loader))
        f_emb = f_emb.squeeze()
        with torch.no_grad():
            query = []
            queryimgs = self.te_dataset.get_query(dataset)
            for queryimg in queryimgs:
                queryimg = queryimg.to(self.device)
                queryemb = self.resnet(queryimg)
                query.append(queryemb)
            q, m, lss1 = self.forward(f_emb, query)

        recalls, medr, meanr = compute_recall(q.cpu(), m.cpu())
        # dict={'dataset':dataset,'emb':m.cpu()}
        # torch_save(self.args.load_path, 'szymodelemb.pt',dict)
        return lss1.item(), recalls, medr, meanr

    def save_model(self, is_max=False, epoch=None, recall=None, medr=None, meanr=None, mse=None):
        print('==> saving models ... ')
        if is_max:
            fname = 'szysaved_model_max_recall_ba58_ba8_lr4.pt'
        else:
            fname = 'szysaved_model_ba58_ba8_lr4.pt'
        torch_save(self.args.load_path, fname, {
            'enc_q': self.enc_q.cpu().state_dict(),
            'enc_m': self.enc_m.cpu().state_dict(),
            # 'predictor': self.predictor.cpu().state_dict(),
            'epoch': epoch,
            'recall': recall,
            'medr': medr,
            'meanr': meanr,
            'mse': mse
        })
        # self.predictor.to(self.device)
        self.enc_q.to(self.device)
        self.enc_m.to(self.device)

    def save_scroes(self):
        f_write(self.args.load_path, f'cross_modal_space_results.txt', {
            'options': vars(self.args),
            'results': self.scores
        })

    def load_model_zoo(self):
        start_time = time.time()
        self.model_zoo = torch_load(self.args.model_zoo)
        print(f"==> {len(self.model_zoo['dataset'])} pairs have been loaded {time.time() - start_time:.2f}s")

    def store_model_embeddings(self):
        print('==> storing model embeddings ... ')
        start_time = time.time()
        # embeddings = {'dataset': [], 'm_emb': [], 'acc': []}
        embeddings={}
        dataset, acc, f_emb = next(iter(self.tr_loader))
        f_emb=f_emb.squeeze()
        # print(len(dataset),len(f_emb))
        # for k in range(self.args.batch_size):
        with torch.no_grad():
            m_emb = self.enc_m(f_emb.to(self.device),self.resemb)
            for k in range(len(dataset)):
                embeddings[dataset[k]]=m_emb[k]
        torch_save(self.args.load_path, 'szymodelemb.pt', embeddings)
        print(f'==> storing embeddings done. ({time.time() - start_time}s)')

    def test(self):
        print(f'Begin test process')
        start_time = time.time()
        self.init_loaders_for_meta_test()
        self.load_encoders_for_meta_test()
        self.load_cross_modal_space()
        self.meta_test()
        print(f'Process done ({time.time() - start_time:.2f})')

    def init_loaders_for_meta_test(self):
        print('==> loading meta-test loaders')
        self.tr_dataset, self.tr_loader = get_meta_test_loader(self.args, mode='train')
        self.te_dataset, self.te_loader = get_meta_test_loader(self.args, mode='test')

    def load_encoders_for_meta_test(self):
        print('==> loading encoders ... ')
        _loaded = torch_load(os.path.join(self.args.load_path,  'szy3saved_model_max_recall_58_21.pt'))
        self.enc_q = QueryEncoder(self.args).to(self.device).eval()
        self.enc_q.load_state_dict(_loaded['enc_q'])
        self.predictor = PerformancePredictor(self.args).to(self.device).eval()
        self.predictor.load_state_dict(_loaded['predictor'])

    def load_cross_modal_space(self):
        print('==> loading the cross modal space ... ')
        self.cross_modal_info = torch_load(os.path.join(self.args.load_path,'retrieval.pt'))
        m_embs = torch.stack(self.cross_modal_info['m_emb']).to(self.device).squeeze()
        self.m_embs=m_embs

    def generate_JLmatrix(self,d, k):
        a = torch.randn([k, d], out=None)
        b = torch.div(torch.tensor(1), torch.sqrt(torch.tensor(k))) * a
        return b

    def JLTransform(self,test, R):
        transformed_matrix = torch.matmul(R, test.t())
        return transformed_matrix

    def compute_recall_JL(self,out_indice):
        # Get query_embs image
        # Compute scores
        index_list = []
        npts = out_indice.shape[0]
        ranks = np.zeros(npts)
        top1 = np.zeros(npts)
        for index in range(npts):
            sorted_index_lst = out_indice[index]
            index_list.append(sorted_index_lst[0])
            # Score
            rank = np.where(sorted_index_lst.cpu()== index)[0][0]
            ranks[index] = rank
            top1[index] = sorted_index_lst[0]
        recalls = {}
        for v in [1, 5, 10, 50, 100]:
            recalls[v] = 100.0 * len(np.where(ranks < v)[0]) / len(ranks)
        return recalls

    def EuclideanDistance(self, t1, t2):
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
