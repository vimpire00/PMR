####################################################################################################

####################################################################################################
import argparse

class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()
    
    def str2bool(self, s):
        return s.lower() in ['true', 't']
   
    def set_arguments(self):
        ##############################################
        self.parser.add_argument('--gpu', type=str, default='0', help='gpus to use, i.e. 0')
        self.parser.add_argument('--device', type=str, default='cpu', help='gpus to use, i.e. 0')
        self.parser.add_argument('--mode', type=str, default='plain_test', help='i.e. train, test')
        self.parser.add_argument('--seed', type=int, default=888, help='seed for reproducibility')
        self.parser.add_argument('--n-dims', type=int, default=128, help='dimension of model and query embedding')
        self.parser.add_argument('--batch_size', type=int, default=58, help='odel and query embedding')
        ##############################################
        self.parser.add_argument('--testdatapath', type=str, default='./test_imgs/mnrtest',  help='path to meta-training or meta-test dataset')
        self.parser.add_argument('--load-path', type=str, default='./PMRModels',
                                 help='base path for loading encoders, cross-modal space, etc.')
        self.parser.add_argument('--model_zoo', type=str, default='./mobemb',
                                 help='base path for loading encoders, cross-modal space, etc.')
        self.parser.add_argument('--n_epochs', type=int, default=5000,
                                 help='base path for loading encoders, cross-modal space, etc.')
        self.parser.add_argument('--model', type=str, default='szysaved_model_max_recall_ba58_ba20_ep10000.pt',
                                 help='base path for loading encoders, cross-modal space, etc.')
        self.parser.add_argument('--hash_path', type=str, default='./modelhash', help='base path')
        self.parser.add_argument('--hash_name', type=str, default='szymob13embs_ba10_da613128.pt', help='base path')
        self.parser.add_argument('--test_imgss', type=str, default='./test_imgss/ai2020f_0_116.pt', help='base path')
        self.parser.add_argument('--picbatch', type=int, default=10, help='base path')
        self.parser.add_argument('--mobmodelspath', type=str, default='./models/mobilemodels', help='base path')

        self.parser.add_argument( "--world_size", type=int, default=2,
             help="The number of parties to launch. Each party acts as its own process"
            )

        self.parser.add_argument( "-s", "--server", type=str, required=False, default='127.0.0.1',
        help="IP address of the server.",
    )
        self.parser.add_argument(
        "-v", "--verbose",
        required=False, default=False, action='store_true',
        help="Log verbosely.",
    )
        self.parser.add_argument('--port', type=str, default='localhost:50051', help='port')
        self.parser.add_argument('--serverport', type=str, default='[::]:50051', help='port')

    def parse(self):
        args, unparsed = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        return args

class Parser2:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()
    
    def str2bool(self, s):
        return s.lower() in ['true', 't']
   
    def set_arguments(self):
        ##############################################
        self.parser.add_argument('--gpu', type=str, default='0', help='gpus to use, i.e. 0')
        self.parser.add_argument('--device', type=str, default='cpu', help='gpus to use, i.e. 0')
        self.parser.add_argument('--mode', type=str, default='plain_test', help='i.e. train, test')
        self.parser.add_argument('--seed', type=int, default=888, help='seed for reproducibility')
        self.parser.add_argument('--n-dims', type=int, default=256, help='dimension of model and query embedding')
        self.parser.add_argument('--batch_size', type=int, default=58, help='odel and query embedding')
        ##############################################
        self.parser.add_argument('--testdatapath', type=str, default='./test_imgs/mnrtest',  help='path to meta-training or meta-test dataset')
        self.parser.add_argument('--load-path', type=str, default='./PMRModels',
                                 help='base path for loading encoders, cross-modal space, etc.')
        self.parser.add_argument('--model_zoo', type=str, default='./mobemb',
                                 help='base path for loading encoders, cross-modal space, etc.')
        self.parser.add_argument('--n_epochs', type=int, default=5000,
                                 help='base path for loading encoders, cross-modal space, etc.')
        self.parser.add_argument('--model', type=str, default='szysaved_model_max_recall_ba58_ba20_emb256_lr4.pt',   help='base path for loading encoders, cross-modal space, etc.')
        self.parser.add_argument('--hash_path', type=str, default='./modelhash', help='base path')
        self.parser.add_argument('--hash_name', type=str, default='szymob13embs_ba10_da613256.pt', help='base path')
        self.parser.add_argument('--test_imgss', type=str, default='./test_imgss/ai2020f_0_116.pt', help='base path')
        self.parser.add_argument('--picbatch', type=int, default=10, help='base path')
        self.parser.add_argument('--mobmodelspath', type=str, default='./models/mobilemodels', help='base path')

        self.parser.add_argument( "-s", "--server", type=str, required=False, default='127.0.0.1',
        help="IP address of the server.",
    )
        self.parser.add_argument(
        "-v", "--verbose",
        required=False, default=False, action='store_true',
        help="Log verbosely.",
    )
        self.parser.add_argument('--port', type=str, default='localhost:50051', help='port')
        self.parser.add_argument('--serverport', type=str, default='[::]:50051', help='port')

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
        self.parser.add_argument('--gpu', type=str, default='0', help='gpus to use, i.e. 0')
        self.parser.add_argument('--device', type=str, default='cpu', help='gpus to use, i.e. 0')
        self.parser.add_argument('--mode', type=str, default='plain_test', help='i.e. train, test')
        self.parser.add_argument('--seed', type=int, default=888, help='seed for reproducibility')
        self.parser.add_argument('--n-dims', type=int, default=512, help='dimension of model and query embedding')
        self.parser.add_argument('--batch_size', type=int, default=58, help='odel and query embedding')
        ##############################################
        self.parser.add_argument('--testdatapath', type=str, default='./test_imgs/mnrtest',  help='path to meta-training or meta-test dataset')
        self.parser.add_argument('--load-path', type=str, default='./PMRModels',
                                 help='base path for loading encoders, cross-modal space, etc.')
        self.parser.add_argument('--model_zoo', type=str, default='./mobemb',
                                 help='base path for loading encoders, cross-modal space, etc.')
        self.parser.add_argument('--n_epochs', type=int, default=5000,
                                 help='base path for loading encoders, cross-modal space, etc.')
        self.parser.add_argument('--model', type=str, default='szysaved_model_max_recall_ba58_ba20_lr4_emb512.pt',
                                 help='base path for loading encoders, cross-modal space, etc.')
        self.parser.add_argument('--hash_path', type=str, default='./modelhash', help='base path')
        self.parser.add_argument('--hash_name', type=str, default='szymob13embs_ba10_da613512.pt', help='base path')
        self.parser.add_argument('--test_imgss', type=str, default='./test_imgss/ai2020f_0_116.pt', help='base path')
        self.parser.add_argument('--picbatch', type=int, default=10, help='base path')
        self.parser.add_argument('--mobmodelspath', type=str, default='./models/mobilemodels', help='base path')

        self.parser.add_argument( "-s", "--server", type=str, required=False, default='127.0.0.1',
        help="IP address of the server.",
    )
        self.parser.add_argument(
        "-v", "--verbose",
        required=False, default=False, action='store_true',
        help="Log verbosely.",
    )
        self.parser.add_argument('--port', type=str, default='localhost:50051', help='port')
        self.parser.add_argument('--serverport', type=str, default='[::]:50051', help='port')

    def parse(self):
        args, unparsed = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        return args

