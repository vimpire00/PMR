####################################################################################################
# TANSmodels: Task-Adaptive Neural Network Search with Meta-Contrastive Learning
# Wonyong Jeong, Hayeon Lee, Geon Park, Eunyoung Hyung, Jinheon Baek, Sung Ju Hwang
# github: https://github.com/wyjeong/TANS, email: wyjeong@kaist.ac.kr
####################################################################################################
import os
import torch.cuda
from parser1 import Parser
import sys
from parse2 import Parser2,Parser3
from parse3 import Parser11,Parser21,Parser31
sys.path.append('./../../')
from misc.utils import *
from retrieval.retrieval import Retrieval
os.environ["CUDA_VISIBLE_DEVICES"]="0"
print(torch.cuda.is_available(),"cuda_ava")
def main(args):
    set_seed(args)
    print(f'mode: {args.mode}')
    if args.mode == 'train':
        # train cross-modal space from model-zoo
        retrieval = Retrieval(args)
        retrieval.train()

    elif args.mode == 'test':
        # test cross-modal space on unseen datasets
        retrieval = Retrieval(args)
        retrieval.evaluate2()

def set_seed(args):
    # Set the random seed for reproducible experiments
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

def set_gpu(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if args.gpu == -1 else args.gpu
    return args

def set_path(args):
    now = datetime.now().strftime("%Y%m%d_%H%M")
    args.log_path = os.path.join(args.base_path, now, 'logs')
    args.check_pt_path = os.path.join(args.base_path, now, 'checkpoints')

    if not os.path.exists(args.base_path):
        os.makedirs(args.base_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.check_pt_path):
        os.makedirs(args.check_pt_path)

    if args.mode == 'train':
        args.retrieval_path = os.path.join(args.base_path, now, 'retrieval')
        if not os.path.exists(args.retrieval_path):
            os.makedirs(args.retrieval_path)

    return args

if __name__ == '__main__':
    # print("train_batch8emb128")
    # main(Parser().parse())
    # print("train_batch8emb256")
    # main(Parser2().parse())
    # print("train_batch8emb512")
    # main(Parser3().parse())
    print("train_batch20")
    print("emb128")
    main(Parser11().parse())
    # print("emb256")
    # main(Parser21().parse())
    # print("emb512")
    # main(Parser31().parse())