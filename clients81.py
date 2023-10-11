# _*_ coding: utf-8 _*_
import grpc
from proto import szy81_pb2
from proto import szy81_pb2_grpc
from google.protobuf import struct_pb2
import argparse
import torch
# Logging
from common import onnx_helper, operation_handler, config
from common import szy_minionn_helpler as minionn_helper
import logging
import logging.config
logging.config.fileConfig('common/logging.conf')
logger = logging.getLogger('minionn')
import time
import random
from c_plain_test import TestC
from parser1 import Parser,Parser2,Parser3
# from tansparse import Parser,Parser2,Parser3
from s_plain_test import TestS
from lib import minionn as minionn
from sys import getsizeof
def prepare_hash_val():
    testS=TestS(args)   
    hash_vals,hash_names=testS.hash_vals,testS.hash_name
    tensors_ws=torch.stack(hash_vals)
    dim=[len(tensors_ws),tensors_ws[0].shape[0],1]
    fractional = 1
    fractional = pow(config.fractional_base, 1)
    name='w'
    values=tensors_ws.flatten().tolist()
    tmp=minionn_helper.put_cpp_tensor(name, values,dim, fractional)
    return tmp,dim
 
#num=58,58*2,58*3
#n=
def main(args,num,n):
    #Create and set up Logger
    random.seed(args.seed)
    loglevel = (logging.DEBUG if args.verbose else logging.INFO)
    logger.setLevel(loglevel)
    logger.info("szy CLIENT")
    testc=TestC(args)
    q_embss=[]
    # First, read the x vector from input convert to list multiply by fractional
    q_datasets=[]
    for i in range(num):
        q_embs,dataset,q_imgs=testc.load_query_data()
        q_embss.extend(q_embs)
        q_datasets.extend(dataset)
    enc_X=[]
    q_datasets=list(q_datasets)
    print("init and gen cpkey/cskey")
    minionn_helper.init(config.SLOTS)
    # minionn_helper.generate_keys(config.client_pkey,config.client_skey)
    # n=q_embss[0].shape[0]
    i=0
    for x in q_embss:
        x_list = [int(config.fractional_base*v) for v in x] 
        enc_X.append(x_list)  
        i=i+1
    # First, read the x vector from input convert to list 
    totaltimes=0
    total_message=0
    jj=0
    score=0 
    max_indexs=[]
    dim=[1,n,1]
    # enc_X=enc_X[:10]
    for x_list in enc_X:       
        jj=jj+1 
        start_time=time.time()
        # With x ready, we can connect to the server to receive the model and w
        channel = grpc.insecure_channel(args.port)
        stub = szy81_pb2_grpc.SZY81Stub(channel)
        response = stub.Precomputation(szy81_pb2.PrecomputationRequest(requesth=True))
        #receive secret keys h0 and encrypted hs enchs
        server_h0 = response.h0
        server_enchs=response.enchs
        #  Use h0 to generate Multiplication triplet<u,v,h0*r> and xs 
        encU = minionn_helper.client_precomputation(server_h0,config.SLOTS,dim)
        input_r = "initial_r0"
        r = minionn_helper.get_cpp_tensor(input_r)
        input_v='v0'
        v= minionn_helper._get_tensor(input_v)
        result_client=minionn_helper.matrix_mult_client(v,dim)
        xs = minionn_helper.vector_sub(x_list,r)
        # send u and xs to server
        result_future = stub.Computation.future(szy81_pb2.ComputationRequest(u=encU,xs=xs))
        result_server = result_future.result().ys
        query_num=int(len(server_enchs)/n)
        b1=torch.zeros((query_num,1),dtype=torch.int64).flatten()
        cpp_b=minionn.VectorInt(b1)
        dims=[query_num,n,1]
        ycs=[]
        for i in range(58):
            yc=result_client[0]
            xc_hs=0
            for j in range(n):
                xc_hs=xc_hs+server_enchs[i*n+j]*r[j]
            yc=yc+int(xc_hs/1000)
            ycs.append(yc)
        max_result=0   
        max_indx=0
        for k in range(query_num):
            temp=result_server[k]+ycs[k]
            if temp>max_result:
                max_result=temp
                max_indx=k
        finish_time = time.time()
        # print(len(encU[0]),len(xs),len(server_enchs),type(server_h0) ,"aaaaaaaaa")
        # print(len(response.enchs),getsizeof(server_enchs[0]),"ench")
        # print(getsizeof(encU[0]),getsizeof(xs),getsizeof(server_h0),getsizeof(response.enchs),"eeeee")
        totaltimes=totaltimes+finish_time - start_time
        bool1=True
        message_size= getsizeof(encU[0])+getsizeof(xs[0])*len(xs)+getsizeof(server_h0[0])+getsizeof(server_enchs[0])*len(server_enchs)+4+8
        ##test_final_result
        response = stub.FinalIndex.future(szy81_pb2.FinalIndexRequest(index=max_indx))
        server_dataset=response.result().ans
        query_dataset=q_datasets[jj-1]
        if query_dataset==server_dataset:
            score=score+1
        total_message=total_message+message_size
    print(score/len(enc_X),score,len(enc_X),"socre")
    print("Processing took " + str(totaltimes) + " seconds.")
    print("Total trial: {} MB".format(total_message/1024/1024))

if __name__ == '__main__':
    # print("emb128 querynum58")
    # main(Parser().parse(),1,128)
    # main(Parser().parse(),5,128)
    # print("emb128 querynum58*10")
    # main(Parser().parse(),10,128)

    # print("emb256 querynum58")
    # main(Parser2().parse(),10,256)
    # main(Parser2().parse(),5,256)
    # main(Parser2().parse(),1,256)

    print("emb512 querynum58")
    main(Parser3().parse(),10,512)
    # main(Parser3().parse(),5,512)
    # main(Parser3().parse(),1,512)