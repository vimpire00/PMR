# _*_ coding: utf-8 _*_
# Logging

import json
import grpc
from proto import szy_pb2
from proto import szy_pb2_grpc
import time
from concurrent import futures
import argparse
import torch
from parser1 import Parser,Parser2,Parser3
import google
from secure_infer import Sec_Infer
import numpy as np
import random
from s_plain_test import TestS
import logging.config
from common import config
# import crypten
import pickle
import tenseal as ts

logging.config.fileConfig('common/logging.conf')
# logger = logging.getLogger('minionn')
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

def gencontext():
    
    context = ts.context(
            scheme=ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=4096,
            plain_modulus=786433,
            coeff_mod_bit_sizes=[40, 20, 40],
            # encryption_type=enc_type,
    )
    # context = ts.context(
    #         scheme=ts.SCHEME_TYPE.CKKS,
    #         poly_modulus_degree=2048,
    #         plain_modulus=786433,
    #         coeff_mod_bit_sizes=[20, 20],
    #         # encryption_type=enc_type,
    # )
    context.generate_relin_keys()
    context.generate_galois_keys()
    context.global_scale = 2 **12
    return context


class SZYServ(object): 
    def __init__(self,hash_val,hash_name,dim,datasets):
        self.w_precomputed = hash_val
        self.dim=dim
        self.query_num=dim[0]
        self.flag=1
        self.ctx=gencontext()
        self.enc_pl=[]
        self.datasets=datasets
    def Precomputation(self, request, context):
    #Precomputation service - returns ONNX client model and ~w
        return szy_pb2.PrecomputationResponse(w=self.w_precomputed,n=self.query_num)

    def Computation(self, request, context):
    #Computation message - receives ~u and x_s and returns y_s
        logger.info("Got computation request.")
        # Perform last precomputation step on U
        xs=minionn_helper.put_cpp_tensor("xs",request.xs, [dim[1],dim[2]])
        decU = minionn_helper.server_decrypt_u(request.u, config.server_skey)
        length=dim[0]*dim[1]*dim[2]
        trueu=decU[:length]
        b1=torch.zeros((self.query_num,1),dtype=torch.int64).flatten()
        cpp_b=minionn.VectorInt(b1)
        u_sum=minionn_helper.extract_sum(trueu, dim,0)
        cpp_w=self.w_precomputed
        cpp_x=xs
        ys=minionn_helper.matrix_mult(cpp_w,cpp_b,u_sum,cpp_x,dim)
        return szy_pb2.ComputationResponse(ys=ys)

    def PreOnlineComputation(self, request, context):
        self.flag=self.flag+1
        self.ctx= ts.context_from(request.pk)        
        y=58
        return szy_pb2.PreOnlineComputationResponse(querynum=y)

    def OnlineComputation(self, request, context):
        print("End To End Computation")
        enc_x =ts.ckks_vector_from(self.ctx, request.x[0])
        results=[]
        nnn=len(self.w_precomputed)
        batchnum=1
        nnn=int(len(self.w_precomputed)/batchnum)
        for i in range(nnn):
            enc_pi=self.w_precomputed[i*batchnum:(i+1)*batchnum].t()
            re=enc_x.mm(enc_pi.cpu())
            results.append(re.serialize())
        enc_pi=self.w_precomputed[(i+1)*batchnum:].t()
        re=enc_x.mm(enc_pi.cpu())
        results.append(re.serialize())
        return szy_pb2.OnlineComputationResponse(y=results)
  
    def FinalIndex(self, request, context):
        fiindex=request.index 
        ans=self.datasets[fiindex]
        return szy_pb2.FinalAnsResponse(ans=ans)
        # return google.protobuf.empty_pb2.Empty()
        
def main(args):

   # set random seed Create and set up Logger
    random.seed(args.seed)
    testS=TestS(args)
    hash_vals,hash_names=testS.hash_vals,testS.hash_name
    #We are now ready for incoming connections. Open the serverzx
    scaled_w=torch.stack(hash_vals)
    scaled_w1=scaled_w*10000
    scaled_w2=scaled_w1[:,:]
    scaled_w=scaled_w1
    datasets=testS.dataset
    channel_options = [
    ('grpc.max_send_message_length', 1024 * 1024 * 1024),
    ('grpc.max_receive_message_length', 1024 * 1024 * 1024),
                ('grpc.enable_retries', 1), 
    ]
    dim=[scaled_w.shape[0],scaled_w.shape[1],1]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),options=channel_options)
    szy_pb2_grpc.add_SZYServicer_to_server(SZYServ(scaled_w,hash_names,dim,datasets), server)
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
    # main(Parser().parse())
    # main(Parser2().parse())
    main(Parser3().parse())