# _*_ coding: utf-8 _*_

import grpc
from proto import szy_pb2
from proto import szy_pb2_grpc
from google.protobuf import struct_pb2
import argparse
import torch
# Logging
from common import config
import logging
import logging.config
import time
import random
from c_plain_test import TestC
from parser1 import Parser,Parser2,Parser3
import torch
import json
import pickle
import tenseal as ts

def gencontext():

    context = ts.context(
            scheme=ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=4096,
            plain_modulus=786433,
            coeff_mod_bit_sizes=[40, 20, 40],
    )

    context.generate_relin_keys()
    context.generate_galois_keys()
    context.global_scale = 2 **12
    return context

def encrypt(context, np_tensor):
    return ts.ckks_tensor(context, np_tensor)

def main(args,ee):
    random.seed(args.seed)
    testc=TestC(args)
    q_embss=[]
    q_datasets=[]
    for i in range(ee):
        q_embs,dataset,q_imgs=testc.load_query_data()
        q_embss.extend(q_embs)
        q_datasets.extend(dataset)
    enc_X=[]
    q_datasets=list(q_datasets)
    print("init and gen cpkey/cskey")
    ctx=gencontext()
    sk=ctx.secret_key()
    message_size1=0
    message_size2=0
    channel_options = [
    ('grpc.max_send_message_length', 1024 * 1024 * 1024),
    ('grpc.max_receive_message_length', 1024 * 1024 * 1024),
                ('grpc.enable_retries', 1), 
    ]
    for x in q_embss:
        bf_vec=ts.ckks_vector(ctx,x)
        enc_x=bf_vec.serialize()
        enc_X.append(enc_x)      
    ser=ctx.serialize(save_public_key=True, save_secret_key=False, save_galois_keys=True, save_relin_keys=True) 
    print("pre_online_trans")
    start_time=time.time()
    pre_start_time=time.time()
    pl_res=[]
    print("online_test")
    j=0
    score=0
    total_time=0
    message_size=0
    channel = grpc.insecure_channel(args.port,options=channel_options)
    stub = szy_pb2_grpc.SZYStub(channel)
    start_time1=time.time()
    response = stub.PreOnlineComputation.future(szy_pb2.PreOnlineComputationRequest(pk=ser))
    query_num=response.result().querynum
    pre_time=time.time()
    for enc_x in enc_X:
        j=j+1
        start_time=time.time()
        # message_size1=8+len(ser)
        result_future = stub.OnlineComputation.future(szy_pb2.OnlineComputationRequest(x=[enc_x]))
        results_server = result_future.result().y
        de_re=[]
        for i in range(len(results_server)):
            result_server=results_server[i]
            deserialized_encrypted_vec = ts.ckks_vector_from(ctx, result_server)
           # 解密密文向量
            decrypted_vec = deserialized_encrypted_vec.decrypt(sk) 
            de_re.extend(decrypted_vec)
        max_index =de_re.index(max(de_re))
        response = stub.FinalIndex.future(szy_pb2.FinalIndexRequest(index=max_index))
        finish_time = time.time()
        message_size2=message_size2+8+len(enc_x)
        server_dataset=response.result().ans
        query_dataset=q_datasets[j-1]
        if query_dataset==server_dataset:
            score=score+1
        total_time=total_time+finish_time-start_time
        message_size=message_size+message_size2
    total_time=total_time+pre_time-start_time1
    print("total_time",total_time)
    print("message_size,MB",message_size/1024/1024)
    print(score/len(q_embss),"acc")

if __name__ == '__main__':
    # ee=1
    # main(Parser().parse(),ee)
    # ee=5
    # main(Parser().parse(),ee)
    # ee=10
    # main(Parser().parse(),ee)
    # ee=1
    # main(Parser2().parse(),ee)
    # ee=10
    # main(Parser2().parse(),ee)
    # ee=5
    # main(Parser2().parse(),ee)
    ee=1
    main(Parser3().parse(),ee)
    ee=5
    main(Parser3().parse(),ee)
    ee=10
    main(Parser3().parse(),ee)