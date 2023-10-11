import numpy as np
import torch
from retrieval.core import *

def compute_recall(query_embs, model_embs):
    # if npts is None:
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
        recalls[v] = len(np.where(ranks < v)[0])
        # recalls[v] = 100.0 * len(np.where(ranks < v)[0]) / len(ranks)
    return recalls[1]

def EuclideanDistance(t1, t2):
    dim = len(t1.shape)
    if dim == 2:
        N, C = t1.shape[0], t1.shape[1]
        M, _ =  t2.shape[0], t2.shape[1]
        dist = -2 * torch.matmul(t1, t2.permute(1, 0))
        dist += torch.sum(t1 ** 2, -1).view(N, 1)
        dist += torch.sum(t2 ** 2, -1).view(1, M)
        return dist
    elif dim == 3:
        B, N, _ = t1.size()
        _, M, _ = t2.size()
        dist = -2 * torch.matmul(t1, t2.permute(0, 2, 1))
        dist += torch.sum(t1 ** 2, -1).view(B, N, 1)
        dist += torch.sum(t2 ** 2, -1).view(B, 1, M)
        return dist
    else:
        print('error...')

class JLCal(object):

   def __init__(self,n,d,k,jltype):
       self.nump =n
       self.dim_init = d
       self.obj_dim = k
       self.jltype=jltype
       self.lamba=0.01
       #type "basic","discrete", "toeplitz"
       self.judge()

   def judge(self):
       n=self.nump
       # log(1/delta)=ou(k*lamba^2)
       k=self.obj_dim
       lamda=self.lamba
       k_upp = torch.div(1, lamda) * torch.div(1, lamda) * torch.log(torch.Tensor([n]))
       delta = k * lamda * lamda
       if torch.tensor(k) > k_upp:
           print("Error")

   def jlt(self,data):
       # return self.r**2 * Circle.pi # 通过实例修改pi的值对面积无影响，这个pi为类属性的值
       return self.jl_transform(data,self.obj_dim,self.jltype)

   def caleudis(self,data1,data2):
       return EuclideanDistance(data1,data2)

   def jl_transform(self,dataset_in, objective_dim, type_transform="basic"):
       """
       This function takes the dataset_in and returns the reduced dataset. The
       output dimension is objective_dim.

       dataset_in -- original dataset, list of Numpy ndarray
       objective_dim -- objective dimension of the reduction
       type_transform -- type of the transformation matrix used.
       If "basic" (default), each component of the transformation matrix
       is taken at random in N(0,1).
       If "discrete", each component of the transformation matrix
       is taken at random in {-1,1}.
       If "circulant", he first row of the transformation matrix
       is taken at random in N(0,1), and each row is obtainedfrom the
       previous one by a one-left shift.
       If "toeplitz", the first row and column of the transformation
       matrix is taken at random in N(0,1), and each diagonal has a
       constant value taken from these first vector.
       """
       if type_transform.lower() == "basic":
           jlt = (1 / math.sqrt(objective_dim)) * np.random.normal(0, 1, size=(objective_dim,
                                                                               len(dataset_in[0])))
       elif type_transform.lower() == "discrete":
           jlt = (1 / math.sqrt(objective_dim)) * np.random.choice([-1, 1],
                                                                   size=(objective_dim, len(dataset_in[0])))
       elif type_transform.lower() == "circulant":
           from scipy.linalg import circulant
           first_row = np.random.normal(0, 1, size=(1, len(dataset_in[0])))
           jlt = ((1 / math.sqrt(objective_dim)) * circulant(first_row))[:objective_dim]
       elif type_transform.lower() == "toeplitz":
           from scipy.linalg import toeplitz
           first_row = np.random.normal(0, 1, size=(1, len(dataset_in[0])))
           first_column = np.random.normal(0, 1, size=(1, objective_dim))
           jlt = ((1 / math.sqrt(objective_dim)) * toeplitz(first_column, first_row))
       else:
           print('Wrong transformation type')
           return None
       trans_dataset = []
       [trans_dataset.append(np.dot(jlt, np.transpose(dataset_in[i])))
        for i in range(len(dataset_in))]
       return trans_dataset

   def jldis(self,data1,data2):
        l=len(data1)
        data=torch.cat([data1,data2],dim=0)
        t_data=self.jlt(data)
        t_data1=t_data[:l]
        t_data2=t_data[l:]
        disjl=self.caleudis(torch.tensor(t_data1),torch.tensor(t_data2))
        return disjl

   def comjlrecall(self,q_emb,m_emb):
       # mmre=torch.matmul(q_emb,m_emb.t())
       # out,in4=torch.sort(mmre,descending=True)
       tr_dist = self.jldis(m_emb, q_emb)
       # print(tr_dist,"tr_dist")
       out, in1 = torch.sort(tr_dist)
       r_1 = 0
       for i in range(len(in1)):
           if in1[i][0] == i:
               r_1 = r_1 + 1
       return r_1/len(in1)