"""
Author: Fritz Alder
Copyright: 
Secure Systems Group, Aalto University 
https://ssg.aalto.fi/

This code is released under Apache 2.0 license
http://www.apache.org/licenses/LICENSE-2.0
"""

# cpp
import cppimport
import cppimport.import_hook
from lib import minionn as minionn

import logging
logger = logging.getLogger('minionn.minionn_helper')

from operator import mul
from functools import reduce

import numpy as np

from . import config

#TODO: Work with tensor objects held by a tensorhandler
class Tensor(object):
    """
    Class for a Tensor that can have a numpy or a cpp reference.
    Contains:
     - Shape
     - CPP representation (optional)
     - Numpy representation (optional but preferred)
    
    When the system gets initialized, all tensors are numpy tensors.
    Whenever a cpp tensor gets requested, it gets generated on demand from the numpy tensor.
    Vice versa, if a numpy tensor is requested but only a cpp vector exists, it also gets generated on demand (e.g. after a MPC operation, only a cpp vector will exist)

    During the computation of the network, only cpp tensors should be used. The main reason for this is that the MPC operations are implemented in cpp with the ABY library and require the tensors to be cpp vectors. Every conversion from cpp to numpy or vice versa contains at least one copy of the whole tensor which should be avoided whenever possible.
    """
    def __init__(self):
        pass

# NOTE: Currently NOT concurrency safe! Only works for a single client
# TODO: Refactor into tensors as objects. Then keep a dict of objects. 
# TODO: Think about splitting minionn helper and matrix stuff
# Maybe node operator can take care of matrices while having 
#  minionn helper only call cpp code
# Or create extra class for matrices and give it to node operator
#  as argument when calling run network

# TODO: Move minionn operations into the actual nodes

# dictionary to hold the cpp tensors mapped by their name
# This is a dict of a tensor name to a tuple of (VectorInt, Shape (as list))
cpp_tensors = {}

def _tensor_is_transposed(name):
    transposed = name.count("T") % 2
    if transposed:
        return True
    else:
        return False

def _tensor_get_base_name(name):
    return name.replace("T","")

def _tensor_normalize_name(name):
    normalized_name = _tensor_get_base_name(name)

    if _tensor_is_transposed(name):
        normalized_name += "T"
    
    return normalized_name

def _has_tensor(name):
    if name in cpp_tensors:
        return True
    return False

def _get_tensor(name):
    name_normalized = _tensor_normalize_name(name)

    if not _has_tensor(name_normalized) \
        or \
        cpp_tensors[name_normalized][0] is None:   
        # Return transposed vector.
        # It does not exist yet, but we can quickly create it
        #  if its transposed exists and has values
        transposed_name = _tensor_normalize_name(name_normalized + "T")
        if _has_tensor(transposed_name) \
        and tensor_has_values(transposed_name):
            _transpose(transposed_name)

    return cpp_tensors[name_normalized][0]
 
def _get_tensor_dim(name):
    """
    Returns the shape (dimension) as a list of the given tensor.
    If name does not exist, we try to return the reversed dimension of 
    its transposed vector
    """
    tname = _tensor_normalize_name(name)
    if _has_tensor(tname):
        return cpp_tensors[tname][1]
    else:  
        # Name does not exist. Try transposed of name
        tname = _tensor_normalize_name(name + "T")
        if _has_tensor(tname):
            return list(reversed(cpp_tensors[tname][1]))
        else:
            logger.error("Cannot get dimension of nonexistent tensor " + name)

def _set_tensor(name, new_tensor, new_dimension):
    cpp_tensors[_tensor_normalize_name(name)] = (new_tensor, new_dimension)
    if new_tensor is not None and config.debug_mode:
        logger.debug("-- Tensor's size is " + str(len(list(new_tensor))))
#        assert + " size:" + str(new_tensor.size())

def _transpose(inp):
    """
    Takes the input vector from cpp_vectors, reshapes it into 
    the dimensions given, transposes the matrix, and creates a new,
    flattened, cpp vector as "<inp>T" with <inp> being the input string.
    """
    # Calculate new name (to prevent double T namings)
    new_name = _tensor_get_base_name(inp) 
    if not _tensor_is_transposed(inp): 
        new_name += "T"
    logger.debug("Transposing " + inp + " to output " + new_name )

    # Get vec and dim
    vec_in = list(cpp_tensors[inp][0])
    dim_in = cpp_tensors[inp][1]

    # Transpose the reshaped matrix
    reshaped = np.reshape(vec_in, dim_in)
    transposed = np.transpose(reshaped)
    dim_out = list(transposed.shape)

    # Flatten and store
    _set_tensor(new_name, minionn.VectorInt(transposed.flatten().tolist()), dim_out)

def put_cpp_tensor(name,values,dimension, fractional = 1):
    if fractional != 1 or not all(isinstance(v, int) for v in values):
        # If fractional is not 1 or we have a list of not solely integers, 
        # perform list comprehension
        tmp = [modulo_pmax(int(fractional * v)) for v in values]    
        tmp1 = minionn.VectorInt(tmp)
        _set_tensor(name, tmp1, dimension)
    else:    
        # Else, simply add to dict
        tmp1=minionn.VectorInt(values)
        _set_tensor(name,tmp1, dimension)
    
    return tmp1


def scale_tensor(name,values, fractional = 1):
    if fractional != 1 or not all(isinstance(v, int) for v in values):
        # If fractional is not 1 or we have a list of not solely integers, 
        # perform list comprehension
        tmp = [modulo_pmax(int(fractional * v)) for v in values]    
        # tmp1 = minionn.VectorInt(tmp)
        # _set_tensor(name, tmp1, dimension)
    return tmp


def get_cpp_tensor(name, reshape = False):
    """
    Returns the cpp tensor associated with name.
    If name ends on T, the transposed tensor is returned.
    If reshape is true, a proper reshaped numpy array is returned
    """
    name_normalized = _tensor_normalize_name(name)

    tensor = list(_get_tensor(name_normalized))

    if reshape:
        # Use numpy to reshape array
        tensor = np.reshape(tensor, _get_tensor_dim(name_normalized))

    return tensor

def get_cpp_tensor_dim(name):
    """
    Returns the shape (dimension) as a list of the given tensor.
    Result is a list
    """
    return _get_tensor_dim(name)
    
def has_cpp_tensor(name):
    """
    Checks if the given named tensor exists.
    Takes the following three cases into account:
     - named vector exists
     - normal vector exists but transposed doesn't ("<name>T")
     - transposed vector "<name>T" exists but named vector doesn't
    """
    if  _has_tensor(_tensor_normalize_name(name)) \
        or _has_tensor(_tensor_get_base_name(name) + "T") \
        or _has_tensor(_tensor_get_base_name(name)):
        return True
    else:
        return False

def tensor_has_values(name):
    """
    Checks if a given tensor, if it exists, has any values or 
    if it just a stub for dimensions.
    """
    if has_cpp_tensor(name) and _get_tensor(name) is not None:
        return True
    else:
        return False

def print_tensor(name):
    normalized = _tensor_normalize_name(name)
    s = normalized
    if normalized != name:
        s += " (aka " + name + ")"
    s += " (dim: " + str(_get_tensor_dim(name)) + ")"
    
    if config.debug_mode:
        s += " (Currently has values: "  
        if tensor_has_values(name):
            s += "Yes. Complete size: " + str(len(get_cpp_tensor(name)))
            s += " First values:" + str(get_cpp_tensor(name)[:config.debug_print_length])
        else:
            s+= "No"

        s += ")"

    return s

def _log_vector_dict():
    logger.debug("Cpp dictionary elements:")
    
    for name in sorted(cpp_tensors):
        logger.debug("  -- " + print_tensor(name))
    logger.debug("End Cpp dictionary")

def copy_tensor(name_src, name_dst):
    src = _tensor_normalize_name(name_src)
    if _has_tensor(src):
        _set_tensor(name_dst, _get_tensor(src), _get_tensor_dim(src) )


""""
MiniONN functions.
These functions are CPP functions and the python functions are just a wrapper
for them.
"""
def init(slots):
    minionn.init(slots)

def init_mpc(ip, port, is_server):
    minionn.init_aby(ip, port, is_server)

def shutdown_mpc():
    minionn.shutdown_aby()

def generate_keys(pkey, skey):
    minionn.gen_keys(pkey, skey)

def server_prepare_w(w_list, pkey):
    """
    Prepares the W to send over to the client.
    This W contains all w from every matrix multiplication
     and is encrypted with the server's public key.
    Arranging the Ws is done doing the following:
     For each m x n * n x o matrix multiplication,
      this multiplication's W has every row of w repeated o times.
     Each multiplication's W is then attached to the overall W.

     Input: 
      - w_list: List of tuples:(name of W, dimensions of matrix multiplication [m,n,o])
      - public key of server
    """

    # We will use numpy to properly arFrange the Ws.
    # In future, this has a way better performance if numpy is
    #  the primary library in use
    overall_w = []
    for (w,dim) in w_list:
        # Get list as reshaped numpy array
        tensor = get_cpp_tensor(w, reshape=False)
        overall_w=tensor
       
    if config.debug_mode:
        logger.info("W has size " + str(len(overall_w)))
        logger.info("W starts with " + str(overall_w[:config.debug_print_length_long]) + " and ends with " + str(overall_w[-config.debug_print_length_long:]))

    return minionn.encrypt_w(minionn.VectorInt(overall_w), pkey)


def server_decrypt_u(encU, skey):
    tmp = minionn.VectorInt([])
    minionn.decrypt_w(encU, skey, tmp)
    return tmp

def modulo_pmax(x_in):
    x_in = x_in % config.PMAX

    if abs(x_in) <= config.PMAX_HALF:
        return x_in
    elif x_in > 0:
        return x_in - config.PMAX
    else:
        return x_in + config.PMAX

def client_precomputation(encW, slot_size,dim):
    """
    Performs the client precomputation.
    This takes the encrypted W from the server and generates
     a v and r for each matrix multiplication.
     r has the shape of x in the W*x multiplication (n x o)
     v has the shape of m x n x o (which gets summed up to n x o later during the client matrix multiplication)
    As the r and v values are needed later, they are stored as r0,v0,r1,v1,.. tensors in the tensor dictionary.
    Input:
     - encrypted W
     - slot size
     - w_list: List of tuples:(name of W, dimensions of matrix multiplication [m,n,o])
    Output:
     - encrypted U that can be sent back to the server
    """   
    logger.info("Started Client Precomputation.")
    m,n,o=dim[0],dim[1],dim[2]
    client_randoms = []
    # Generate v    
    v = np.random.randint(config.PMAX, dtype='uint64', size = (m, n, o))
    r = np.random.randint(config.PMAX, dtype='uint64', size = (o, n))
    client_randoms.append((r,v))
    logger.info(" - Generated r and v values:")
    # Now assemble the big r and v that are used for precomputation
    assembled_R = []
    assembled_V = []
    # Assemble R by repeating r_i for every row of W (m times)
    for dm in range(0, m): # For every server row (m) (W row)
        assembled_R.extend(client_randoms[0][0][0].tolist()) # Append a row of r (here, column because it is transposed - Matrix multiplication takes a row times a column)
    # Assemble v by just appending all v's after each other
    assembled_V.extend(client_randoms[0][1].flatten().tolist())
    # Now we need to transpose the r matrices so that they can be used later (remember, we used r as columns earlier for the matrix multiplication with W)
    logger.info(" - Transposing r values:")
    for i in range(0,len(client_randoms)):
        # Transpose r
        client_randoms[i] = (client_randoms[i][0].T, client_randoms[i][1])
        # And convert the uint numpy arrays to int cpp arrays for later use
        #  NOTE: We use a modulo with PMAX here to convert from uint to int
        #  This is the same that is done on the cpp side for the homomorphic encryptions.
        #  For the precomputation, Uint64 is needed, and for everything afterwards, int64
        iR = minionn.VectorInt([modulo_pmax(r) for r in client_randoms[i][0].flatten().tolist()])
        _set_tensor("initial_r" + str(i), iR, list(client_randoms[i][0].shape))

        iV = minionn.VectorInt([modulo_pmax(v) for v in client_randoms[i][1].flatten().tolist()])
        _set_tensor("v" + str(i), iV, list(client_randoms[i][1].shape))
  
    # Generate assembled uint vectors
    uR = minionn.VectorUInt(assembled_R)
    uV = minionn.VectorUInt(assembled_V)
    # Use them for the client precomputation
    encU = minionn.client_precomputation(encW, uR, uV)
    logger.info("Client Precomputation success.")

    return encU
    
def client_precomputation3(encW, slot_size,dim):
    logger.info("Started Client Precomputation.")
    m,n,o=dim[0],dim[1],dim[2]
    client_randoms = []
    # Generate v
    v1 = np.random.randint(config.PMAX, dtype='uint64', size = (1, n, o))
    v =np.tile(v1,(m,1,1))
    r = np.random.randint(config.PMAX, dtype='uint64', size = (o, n))
    client_randoms.append((r,v))
    logger.info(" - Generated r and v values:")
    # Now assemble the big r and v that are used for precomputation
    assembled_R = []
    assembled_V = []
    for dm in range(0, m): # For every server row (m) (W row)
        # for do in range(0, o): # For every client column o (x col)
        assembled_R.extend(client_randoms[0][0][0].tolist()) # Append a row of r (here, column because it is 
    assembled_V.extend(client_randoms[0][1].flatten().tolist())
    logger.info(" - Transposing r values:")
    for i in range(0,len(client_randoms)):
        client_randoms[i] = (client_randoms[i][0].T, client_randoms[i][1])
        iR = minionn.VectorInt([modulo_pmax(r) for r in client_randoms[i][0].flatten().tolist()])
        _set_tensor("initial_r" + str(i), iR, list(client_randoms[i][0].shape))
        iV = minionn.VectorInt([modulo_pmax(v) for v in v1.flatten().tolist()])
        _set_tensor("v" + str(i), iV, list(client_randoms[i][1].shape))

    # Generate assembled uint vectors
    uR = minionn.VectorUInt(assembled_R)
    uV = minionn.VectorUInt(assembled_V)
    # Use them for the client precomputation
    encU = minionn.client_precomputation(encW, uR, uV)
    logger.info("Client Precomputation success.")

    return encU


def client_chamlon(enc_a1, ass_a0,ass_a2):
    logger.info("Started Client Precomputation.")
    client_randoms = []
    uR = minionn.VectorUInt(ass_a0)
    uV = minionn.VectorUInt(ass_a2)
    # Use them for the client precomputation
    encU = minionn.client_precomputation(enc_a1, uR, uV)
    logger.info("Client Precomputation success.")

    return encU


def extract_sum(inp, dimensions, offset):
    """
    Extracts the sum of the tensor of shape dimension (beginning
    at offset) and returns it.
    dim is assuming a list for [m, n, o] for the matrix calculation mxn * nxo
    This is equal to crow, ccol, srow where server matrix gets multiplied with client matrix
    """
    tmp = minionn.VectorInt([])
    minionn.extract_sum(inp, tmp, 
            dimensions[1], dimensions[2], dimensions[0], 
            offset)

    logger.debug("Extract sum: Extracted with offset " + str(offset)+ " and dimensions " + str(dimensions))
    if config.debug_mode:
        logger.debug("Extracted U starts with " + str(list(tmp)[:config.debug_print_length_long]) + " and ends with " + str(list(tmp)[-config.debug_print_length_long:]))
    return tmp

def vector_add(vec_a, vec_b):
    cpp_a = minionn.VectorInt(vec_a)
    cpp_b = minionn.VectorInt(vec_b)
    return minionn.vector_add(cpp_a, cpp_b)

def vector_sub(vec_a, vec_b):
    cpp_a = minionn.VectorInt(vec_a)
    cpp_b = minionn.VectorInt(vec_b)
    return minionn.vector_sub(cpp_a, cpp_b)

def vector_floor(vector):
    minionn.vector_floor(vector, config.fractional_base)


def matrix_mult(w,b,u,x, dims):
    """
    calculates W*x + U + b or
    if order_w_x is False, calculates (W' * X' + U)' + b
    """
    tmp = minionn.VectorInt([])
    cpp_w=w
    cpp_b=b
    cpp_x=x
    instance_u=u
    minionn.matrixmul(cpp_w,cpp_b,instance_u,cpp_x,dims[1],dims[2],dims[0],tmp)
    minionn.vector_floor(tmp, pow(config.fractional_base, 1) * config.fractional_downscale)
    return tmp


def matrix_mult_client(cpp_v,dims):
    tmp = minionn.VectorInt([])
    #Compute and store
    minionn.matrixmul_simple(cpp_v,dims[1],dims[2],dims[0],tmp)
    # Floor the resulting vector to reverse the fractional shifting
    minionn.vector_floor(tmp, pow(config.fractional_base,1) * config.fractional_downscale)
    return tmp

def matrix_mult_client4(cpp_v,num_v):
    tmp = minionn.VectorInt([])
    #Compute and store
    minionn.sum_v(cpp_v,num_v,tmp)
    # Floor the resulting vector to reverse the fractional shifting
    minionn.vector_floor(tmp, pow(config.fractional_base,1) * config.fractional_downscale)
    return tmp

def online_precomputation(encW, slot_size,dim,h):
    """
    Input:
     - encrypted W  v(m*n*o)   r(n*o)
     - w_list: List of tuples:(name of W, dimensions of matrix multiplication [m,n,o])
    """
    logger.info("Started Client Precomputation.")
    m,n,o=dim[0],dim[1],dim[2]
    client_randoms = []
    tmp = [modulo_pmax(int(config.fractional_base * v)) for v in h[0]]    
    r=np.array(tmp,dtype='uint64')
    # Generate v
    v = np.zeros((m,n,o), dtype='uint64')
    client_randoms.append((r,v))
    logger.info(" - Generated r and v values:")
    # Now assemble the big r and v that are used for precomputation
    assembled_R = []
    assembled_V = []
    for dm in range(0, m): # For every server row (m) (W row)
        assembled_R.extend(client_randoms[0][0].tolist()) # Append a row of r (here, column because it is transposed - Matrix multiplication takes a row times a column)
    # Assemble v by just appending all v's after each other
    assembled_V.extend(client_randoms[0][1].flatten().tolist())
    logger.info(" - Transposing r values:")
    for i in range(0,len(client_randoms)):
        # Transpose r
        client_randoms[i] = (client_randoms[i][0].T, client_randoms[i][1])
    uR = minionn.VectorUInt(assembled_R)
    uV = minionn.VectorUInt(assembled_V)

    encU = minionn.client_precomputation(encW, uR, uV)
    print(len(encU),'len_enc_u')
    logger.info("Client Precomputation success.")

    return encU


def online_x_mulw(encx, slot_size,dim,w):
    #dim[1,128,58]
    # Use numpy to generate r and v
    client_randoms = []
    # Generate v    
    v = np.ones([dim[0], dim[1], dim[2]],dtype='uint64')
    r=w
    client_randoms.append((r,v))
    # Now assemble the big r and v that are used for precomputation
    assembled_V = v.flatten().tolist()
    # Generate assembled uint vectors
    w = np.array([1,2,3,4,5,6,7,8],dtype='uint64')
    uR = minionn.VectorUInt(w)
    print(uR,"uR")
    uV = minionn.VectorUInt(assembled_V)
    # Use them for the client precomputation
    encU = minionn.client_precomputation_szy(encx, uR, uV)
    logger.info("Client Precomputation success.")

    return encU