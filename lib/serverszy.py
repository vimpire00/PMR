"""
Author: Fritz Alder
Copyright: 
Secure Systems Group, Aalto University 
https://ssg.aalto.fi/

This code is released under Apache 2.0 license
http://www.apache.org/licenses/LICENSE-2.0
"""

import argparse
from concurrent import futures
import sys, os, time
from operator import mul
from functools import reduce
# cpp
import cppimport
import cppimport.import_hook
# cppimport.set_quiet(True)
import torch
#onnx+
import onnx
#gRPC for client-server communication
import grpc
#project imports
# from common import minionn_onnx_pb2_grpc as minionn_grpc
# from common import minionn_onnx_pb2
from common import onnx_helper, minionn_helper, operation_handler, config
# Logging
import logging
import logging.config
import minionn as m
logging.config.fileConfig('common/logging.conf')
logger = logging.getLogger('minionn')
# import minionn as m

class MinioNNServicer(object):
  """
  The service definition for GRPC.
  """
  def __init__(self, model, w, nodes, ip, mpc_port):
    self.model_client = model
    self.w_precomputed = w
    self.nodes = nodes
    self.ip = ip
    self.mpc_port = mpc_port

  def Precomputation(self, request, context):
    """
    Precomputation service - returns ONNX client model and ~w
    """
    logger.info("Got precomputation request. Responding...")
    return minionn_onnx_pb2.PrecomputationResponse(model=self.model_client, w=self.w_precomputed)

  def Computation(self, request, context):
    """
    Computation message - receives ~u and x_s and returns y_s
    """
    logger.info("Got computation request.")
    logger.debug("xs has length: " + str((len(request.xs))))

    logger.info("Opening MPC server port. Waiting for client to connect...")
    minionn_helper.init_mpc(self.ip, self.mpc_port, True)

    # Perform last precomputation step on U
    decU = minionn_helper.server_decrypt_u(request.u, config.server_skey)
    logger.debug("U has length: " + str(len(list(decU))) )

    # Now system is ready to start NN
    handler = operation_handler.OperationHandler(self.nodes, self.model_client.graph.input[0].name)
    handler.init_server(decU)
    logger.info("Calculate_result.")
    result = minionn_helper.vector_add(decU,decU)
    logger.info("Shutting down MPC server again.")
    minionn_helper.shutdown_mpc()
    logger.info("Computation response:" + str(result))

    return minionn_onnx_pb2.ComputationResponse(ys=result)

def main():
    parser = argparse.ArgumentParser(description="MiniONN - ONNX compatible version")
    parser.add_argument(
        "-i","--input",
        type=str, required=True,
        help="The input protobuf file.",
    )
    parser.add_argument(
        "-p","--port",
        type=int, required=False, default=config.port_rpc,
        help="Server port.",
    )
    parser.add_argument(
        "-m","--mpc_port",
        type=int, required=False, default=config.port_aby,
        help="Server port for MPC.",
    )
    parser.add_argument(
        "-v", "--verbose",
        required=False, default=False, action='store_true',
        help="Log verbosely.",
    )

    args = parser.parse_args()

    """
    Create and set up Logger
    """
    loglevel = (logging.DEBUG if args.verbose else logging.INFO)
    logger.setLevel(loglevel)
    logger.info("MiniONN SERVER")

    """
    First, read the model from input and strip it down for client
    """

    """
    With the two models loaded, we now prepare the model for local Computation
    This includes:
        - loading tensors from model as python lists
        - loading the model to C++
        - generating key
        - precomputing ~w
    """
    logger.info("Parsing model into python and C++... NON")

    # Get tensors and dimensions from onnx
    # tensors = onnx_helper.retrieveTensorsFromModel(model)
    # tensors_dims = onnx_helper.retrieveTensorDimensionsFromModel(model)
    # print(tensors_dims,"tensor_dim")
    # Get nodes from model parse it for ws and bs
    # nodes = onnx_helper.retrieveNodesFromModel(model)
    # tensors_b, tensors_w = onnx_helper.get_bs_and_ws(nodes, tensors)
    # tensors_w =torch.tensor([3,2])
    tensors_w=['1']
    logger.debug("Retrieved w:")
    logger.debug("ws are:" + str(tensors_w))
    # logger.debug("bs are:" + str(tensors_b))

    # Do a sanity #健全的test on the detected Ws
    # If a W gets reshaped before being used, we would not detect it 
    # as an input to a Gemm
    # NOTE: This might be a problem in the future
    # assert len(tensors_w) == len(tensors_b), "Not all W matrices detected! Do some Ws change before being used? (e.g. reshape)"

    # Put tensors into cpp vector dict
    # We use fractions to shift the tensors from floats to integers
    # This means we multiply every w and b with a fraction
    #  The w gets mutliplied with the fractional
    #  The b gets multiplied with the fractional*fractional
    #  This is because the client also multiplies his input with the fractional and 
    #   W*x results in fractional*fractional for b

    # Iterate over tensor dimensions because there might be tensors
    #  that do not exist yet (have no tensor entry) but whose dimension is known
    p_embs=torch.tensor([3,2])
    dim=p_embs.shape
    num_emb=p_embs.shape[0]
    for i in range(num_emb):
        name=str(i)
        value=p_embs
        fractional = 1
        # Get value that belongs to this dim
        # It might not exist, then the dimension is an output or input
        # Keep the value at None then but still register it
        # value = None
        # if name in tensors:
        #     value = tensors[name]
       
        # Adjust the fractional for bs (see above)
        # if name in tensors_b:
        #     fractional = pow(config.fractional_base, 2)
         
        # Same for w
        if name in tensors_w:
            fractional = pow(config.fractional_base, 1)
            print(name,fractional,"fractional")
        # And call put
        minionn_helper.put_cpp_tensor(name, value, dim, fractional)

    logger.info("... parsing model done.")

    logger.info("Calculatung ~w")
  
    # First, generate keys
    if not os.path.exists(config.asset_folder):
        os.makedirs(config.asset_folder)
        logger.info("Created directory " + config.asset_folder)

    logger.info("Generating keys")
    minionn_helper.init(config.SLOTS)
    minionn_helper.generate_keys(config.server_pkey,config.server_skey)
    # Prepare w for precomputation
    # For this, first create a NodeOperator stub that parses the model
    #  and can give us the list of Ws (already transposed etc)
    # The minionn helper then puts together the w correctly
    # logger.info("Parsing network")
    # print(model.graph.input,"names")
    # tmp = operation_handler.OperationHandler(nodes, model.graph.input[0].name, simulation=True)
    # logger.info("Performing precomputation on W")
    origin_w =minionn.VectorInt([1,2,3,4])
    encW=minionn.encrypt_w(minionn.VectorInt(origin_w), pkey)  
    # w = minionn_helper.server_prepare_w(tmp.get_w_list(), config.server_pkey)
    w = m.VectorInt([])
    m.decrypt_w(encW, config.server_pkey,w)
    """
    We are now ready for incoming connections. Open the server
    """
    logger.info("Done with server precomputations. Starting server.")
    servicer = MinioNNServicer(model_client, w, nodes, config.ip, args.mpc_port)
    
    logger.info("Starting to listen on port " + str(args.port))
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=config.grpc_options)
    # minionn_grpc.add_MinioNNServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:' + str(args.port))
    server.start()

    _ONE_DAY_IN_SECONDS = 60 * 60 * 24
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    main()
