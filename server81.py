# _*_ coding: utf-8 _*_

from common import  operation_handler, config
from common import szy_minionn_helpler as minionn_helper
# Logging
import logging
import json
import logging.config
logging.config.fileConfig('common/logging.conf')
logger = logging.getLogger('minionn')
import grpc
from proto import szy81_pb2
from proto import szy81_pb2_grpc
import time
from concurrent import futures
import argparse
import torch
from lib import minionn as minionn
import time
import random
from s_plain_test import TestS
from parser1 import Parser,Parser2,Parser3
# from tansparse import Parser,Parser2,Parser3
import google

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class SZYServ(object):
    def __init__(self, ench0,h0,enchs,w,dim,datasets,n):
        self.ench0=ench0
        self.dim=dim
        self.query_num=dim[0]
        self.datasets=datasets
        self.enchs=enchs
        self.hdim=[1,n,1]
        self.h0=h0
        self.hs=w
        self.n=n
    # 重写父类方法，返回消息
    def Precomputation(self, request, context):
    #Precomputation service - returns ONNX client model and ~w
        logger.info("Got precomputation request. Responding...")
        return szy81_pb2.PrecomputationResponse(h0=self.ench0,enchs=self.enchs)
        
    def Computation(self, request, context):
    #Computation message - receives ~h0 and x_s and returns y_s
        logger.info("Got computation request.")
        # Perform last precomputation step on U
        decU = minionn_helper.server_decrypt_u(request.u, config.server_skey)
        length=self.hdim[0]*self.hdim[1]*self.hdim[2]
        trueu=decU[:length]
        u_sum=minionn_helper.extract_sum(trueu, self.hdim,0)
        cpp_w=self.h0
        xs=minionn_helper.put_cpp_tensor("xs",request.xs, [self.hdim[1],self.hdim[2]])
        cpp_x=xs
        b1=torch.zeros((1,1),dtype=torch.int64).flatten()
        cpp_b=minionn.VectorInt(b1)
        ys_h0=minionn_helper.matrix_mult(cpp_w,cpp_b,u_sum,cpp_x,self.hdim)
        ys_hs=[]
        for i in range(self.query_num):
            ys_h=0
            for j in range(self.n):
                ys_hs_j=self.enchs[i*self.n+j]*cpp_x[j]
                ys_h=ys_h+ys_hs_j
            ys_hs.append(int(ys_h/1000)+ys_h0[0])
        return szy81_pb2.ComputationResponse(ys=ys_hs)

    def FinalIndex(self, request, context):
        fiindex=request.index 
        ans=self.datasets[fiindex]
        return szy81_pb2.FinalAnsResponse(ans=ans)

def main(args,n):
   # set random seed Create and set up Logger
    random.seed(args.seed)
    print("Preparing w into python and C++...")
    testS=TestS(args)
    datasets=testS.dataset
    h0 = torch.randint(-9999, 9999, (n,),dtype=torch.int64,device='cuda:0').flatten()
    def prepare_w():
        enchashvals=[]
        hash_vals,hash_names=testS.hash_vals,testS.hash_name
        tensors_ws=torch.stack(hash_vals)
        tensors_ws=hash_vals
        tensors_ws=torch.stack(hash_vals)
        dim=[len(hash_vals),n,1]
        fractional = 1
        fractional = pow(config.fractional_base, 1)
        name='w'
        values=tensors_ws.flatten().tolist()
        tmp=minionn_helper.put_cpp_tensor('w', values,dim, fractional)
        for hash_val in hash_vals:
            ench=hash_val-h0
            ench=minionn_helper.scale_tensor('hs_h0',ench, fractional)
            enchashvals.extend(ench)

        return enchashvals,tmp,dim
     
    enchs,tensors_ws,dim=prepare_w()

    def prepare_h0():
        fractional = 1
        fractional = pow(config.fractional_base, 1)
        name='h0'
        values=h0.flatten().tolist()
        tmph0=minionn_helper.scale_tensor(name, values,fractional)
        return tmph0

    h0=prepare_h0()
    print("... preparing w and h0 done.")
    print("Generating keys")
    minionn_helper.init(config.SLOTS)
    minionn_helper.generate_keys(config.server_pkey,config.server_skey)
    ench0=minionn.encrypt_w(minionn.VectorInt(h0), config.server_pkey)
    #We are now ready for incoming connections. Open the serverzx
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    szy81_pb2_grpc.add_SZY81Servicer_to_server(SZYServ(ench0,minionn.VectorInt(h0),enchs,tensors_ws,dim,datasets,n), server)
    port_number=args.serverport
    server.add_insecure_port(port_number)
    server.start()
    print('server start...')
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    # print("query_emb=128")
    # main(Parser().parse(),128)
    # print("query_emb=256")
    # main(Parser2().parse(),256)
    print("query_emb=512")
    main(Parser3().parse(),512)