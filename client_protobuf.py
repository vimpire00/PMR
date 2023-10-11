# _*_ coding: utf-8 _*_

import grpc
from proto import szy_pb2
from proto import szy_pb2_grpc
from google.protobuf import struct_pb2
import argparse
import torch
# Logging
from common import config
from common import szy_minionn_helpler as minionn_helper
import logging
import logging.config
import time
import random
from c_plain_test import TestC
from parser1 import Parser
from lib import minionn as m

def main(args):
    #Create and set up Logger
    random.seed(args.seed)
    testc=TestC(args)
    q_embss,dataset,q_imgs=testc.load_query_data()
    # First, read the x vector from input convert to list multiply by fractional
    enc_X=[]
    print("init and gen cpkey/cskey")
    # minionn_helper.init(config.SLOTS)
    # minionn_helper.generate_keys(config.client_pkey,config.client_skey)
    start_time=time.time()

    for x in q_embss:
        x_list = [int(config.fractional_base*v) for v in x] 
        enc_x=m.encrypt_w(m.VectorInt(x_list), config.client_pkey)
        enc_X.append(enc_x)  

    for enc_x in enc_X:
        print("Successfuly read X and fractional from input.")
        # With x ready, we can connect to the server to receive the model and w
        channel = grpc.insecure_channel(args.port)
        stub = szy_pb2_grpc.SZYStub(channel)
        #GEN KEYS 
        result_future = stub.OnlineComputation.future(szy_pb2.OnlineComputationRequest(x=enc_x))
        # Connect to MPC port  
        print("Establishing MPC connection")
        # Get server result and calculate final result
        result_server = result_future.result().y
    finish_time = time.time()
    print("Processing took " +- str(finish_time - start_time) + " seconds.")



if __name__ == '__main__':
    main(Parser().parse())