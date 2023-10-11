####################################################################################################
# TANSmodels: Task-Adaptive Neural Network Search with Meta-Contrastive Learning
# Wonyong Jeong, Hayeon Lee, Geon Park, Eunyoung Hyung, Jinheon Baek, Sung Ju Hwang
# github: https://github.com/wyjeong/TANS, email: wyjeong@kaist.ac.kr
####################################################################################################
import argparse

class Parser2:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()

    def str2bool(self, s):
        return s.lower() in ['true', 't']

    def set_arguments(self):
        ##############################################
        self.parser.add_argument('--device', type=str, default='cuda:0', help='gpus to use, i.e. 0')
        self.parser.add_argument('--mode', type=str, default='test', help='i.e. train, test')
        self.parser.add_argument('--seed', type=int, default=888, help='seed for reproducibility')
        self.parser.add_argument('--batch-size', type=int, default=58, help='batch size')
        self.parser.add_argument('--n-groups', type=int, default=140, help='number of meta-training datasets')
        self.parser.add_argument('--n-dims', type=int, default=256, help='dimension of model and query embedding')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        ##############################################
        self.parser.add_argument('--model_zoo', type=str, default='./../szyretrival/data/mobemb11',
                                 help='path to meta-training or meta-test dataset')
        # self.parser.add_argument('--traindatapath', type=str, default='./datasets/mnrtestdataresemb',
        self.parser.add_argument('--traindatapath', type=str, default='./../szyretrival/data/tar/mnrtrain',
                                 help='path to meta-training or meta-test dataset')
        self.parser.add_argument('--testdatapath', type=str, default='./../szyretrival/data/tar/mnrtest',
                                 help='path to meta-training or meta-test dataset')
        self.parser.add_argument('--base-path', type=str, default='./datasets/log',
                                 help='base parent path for logging, saving, etc.')
        self.parser.add_argument('--load-path', type=str, default='./',
                                 help='base path for loading encoders, cross-modal space, etc.')
        # self.parser.add_argument('--load-model', type=str, default=None,
        self.parser.add_argument('--load-model', type=str, default='szysaved_model_max_recall_ba58_ba8_lr4_emb256_ep20000.pt',
        # self.parser.add_argument('--load-model', type=str, default='szysaved_model_ba58_ba20_ep11000.pt',
        # self.parser.add_argument('--load-model', type=str, default='szysaved_model_ba58_ba8_mmd_con_lr4.pt',
                                 help='base path for loading encoders, cross-modal space, etc.')
        self.parser.add_argument('--n_epochs', type=int, default=10000,
                                 help='base path for loading encoders, cross-modal space, etc.')

    def parse(self):
        args, unparsed = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        return args

class Parser3:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()

    def str2bool(self, s):
        return s.lower() in ['true', 't']

    def set_arguments(self):
        ##############################################
        self.parser.add_argument('--device', type=str, default='cuda:0', help='gpus to use, i.e. 0')
        self.parser.add_argument('--mode', type=str, default='test', help='i.e. train, test')
        self.parser.add_argument('--seed', type=int, default=888, help='seed for reproducibility')
        self.parser.add_argument('--batch-size', type=int, default=58, help='batch size')
        self.parser.add_argument('--n-groups', type=int, default=140, help='number of meta-training datasets')
        self.parser.add_argument('--n-dims', type=int, default=512, help='dimension of model and query embedding')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        ##############################################
        self.parser.add_argument('--model_zoo', type=str, default='./../szyretrival/data/mobemb11',
                                 help='path to meta-training or meta-test dataset')
        # self.parser.add_argument('--traindatapath', type=str, default='./datasets/mnrtestdataresemb',
        self.parser.add_argument('--traindatapath', type=str, default='./../szyretrival/data/tar/mnrtrain',
                                 help='path to meta-training or meta-test dataset')
        self.parser.add_argument('--testdatapath', type=str, default='./../szyretrival/data/tar/mnrtest',
                                 help='path to meta-training or meta-test dataset')
        self.parser.add_argument('--base-path', type=str, default='./datasets/log',
                                 help='base parent path for logging, saving, etc.')
        self.parser.add_argument('--load-path', type=str, default='./',
                                 help='base path for loading encoders, cross-modal space, etc.')
        # self.parser.add_argument('--load-model', type=str, default=None,
        self.parser.add_argument('--load-model', type=str, default='szysaved_model_max_recall_ba58_ba8_lr4__emb512_ep20000.pt',
        # self.parser.add_argument('--load-model', type=str, default='szysaved_model_ba58_ba20_ep11000.pt',
        # self.parser.add_argument('--load-model', type=str, default='szysaved_model_ba58_ba8_mmd_con_lr4.pt',
                                 help='base path for loading encoders, cross-modal space, etc.')
        self.parser.add_argument('--n_epochs', type=int, default=10000,
                                 help='base path for loading encoders, cross-modal space, etc.')

    def parse(self):
        args, unparsed = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        return args

