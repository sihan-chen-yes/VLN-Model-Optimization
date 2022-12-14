import argparse
import os
import torch
#import nni

class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")

        # General
        self.parser.add_argument('--iters', type=int, default=20000)
        self.parser.add_argument('--name', type=str, default='default')
        self.parser.add_argument('--train', type=str, default='speaker')
        self.parser.add_argument('--task',type=str,default='R2R')
        # Data preparation
        self.parser.add_argument('--maxInput', type=int, default=77, help="max input instruction")
        self.parser.add_argument('--maxDecode', type=int, default=120, help="max input instruction")
        self.parser.add_argument('--maxAction', type=int, default=20, help='Max Action sequence')
        self.parser.add_argument('--batchSize', type=int, default=64)
        self.parser.add_argument('--ignoreid', type=int, default=-100)
        self.parser.add_argument('--feature_size', type=int, default=2048)
        self.parser.add_argument("--loadOptim",action="store_const", default=False, const=True)

        # Load the model from
        self.parser.add_argument("--speaker", default=None)
        self.parser.add_argument("--listener", default=None)
        self.parser.add_argument("--load", type=str, default=None)

        # More Paths from
        self.parser.add_argument("--aug", default=None)
        self.parser.add_argument('--train_sub',action='store_true',default=False)


        # Listener Model Config
        self.parser.add_argument("--zeroInit", dest='zero_init', action='store_const', default=False, const=True)
        self.parser.add_argument("--mlWeight", dest='mlWeight', type=float, default=0.05)
        self.parser.add_argument("--teacherWeight", dest='teacher_weight', type=float, default=1.)
        self.parser.add_argument("--accumulateGrad", dest='accumulate_grad', action='store_const', default=False, const=True)
        self.parser.add_argument("--features", type=str, default='imagenet')
        self.parser.add_argument("--zero_feature", dest="zero_feature", action="store_const", default=False, const=True)
        self.parser.add_argument('--debug',action='store_true')
        # Env Dropout Param
        self.parser.add_argument('--featdropout', type=float, default=0.3)

        # SSL configuration
        self.parser.add_argument("--selfTrain", dest='self_train', action='store_const', default=False, const=True)

        # Submision configuration
        self.parser.add_argument("--candidates", type=int, default=1)
        self.parser.add_argument("--paramSearch", dest='param_search', action='store_const', default=False, const=True)
        self.parser.add_argument("--submit", action='store_const', default=False, const=True)
        self.parser.add_argument("--beam", action="store_const", default=False, const=True)
        self.parser.add_argument("--alpha", type=float, default=0.5)

        # Training Configurations
        self.parser.add_argument('--optim', type=str, default='rms')    # rms, adam
        self.parser.add_argument('--lr', type=float, default=0.0001, help="The learning rate")
        self.parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
        self.parser.add_argument('--dropout', type=float, default=0.5)
        self.parser.add_argument('--feedback', type=str, default='sample',
                            help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``')
        self.parser.add_argument('--teacher', type=str, default='final',
                            help="How to get supervision. one of ``next`` and ``final`` ")
        self.parser.add_argument('--epsilon', type=float, default=0.1)
        self.parser.add_argument('--static_gcn_weights',action='store_true',default=False)
        self.parser.add_argument('--static_gcn_weights_only',action='store_true',default=False)

        # Model hyper params:
        self.parser.add_argument('--rnnDim', dest="rnn_dim", type=int, default=512)
        self.parser.add_argument('--wemb', type=int, default=256)
        self.parser.add_argument('--aemb', type=int, default=64)
        self.parser.add_argument('--proj', type=int, default=512)
        self.parser.add_argument("--fast", dest="fast_train", action="store_const", default=False, const=True)
        self.parser.add_argument("--valid", action="store_const", default=False, const=True)
        self.parser.add_argument("--candidate", dest="candidate_mask",
                                 action="store_const", default=False, const=True)

        self.parser.add_argument("--bidir", type=bool, default=True)    # This is not full option
        self.parser.add_argument("--encode", type=str, default="word")  # sub, word, sub_ctx
        self.parser.add_argument("--subout", dest="sub_out", type=str, default="tanh")  # tanh, max
        self.parser.add_argument("--attn", type=str, default="soft")    # soft, mono, shift, dis_shift

        self.parser.add_argument("--angle_feat", dest="angle_feat", action="store_const", default=False, const=True)
        self.parser.add_argument("--angle_feat_size", dest="angle_feat_size", type=int, default=128)
        self.parser.add_argument("--visual_feat", dest="visual_feat", action="store_const", default=False, const=True)
        self.parser.add_argument("--visual_feat_size", dest="visual_feat_size", type=int, default=512)
        self.parser.add_argument("--out_feats",type=int,default=300)
        self.parser.add_argument("--gcn_dim",type=int,default=100)
        self.parser.add_argument("--egcn_activation",type=str,default='relu')
        self.parser.add_argument("--distance_decay_function",type=str,default='exp')
        self.parser.add_argument("--CLIP_language",action='store_true',default=False)
        self.parser.add_argument("--top_N_obj", dest="top_N_obj", type=int, default=8)
        self.parser.add_argument("--glove_dim", dest='glove_dim', type=int, default=300)
        self.parser.add_argument("--clip_dim", dest='clip_dim', type=int, default=512)
        self.parser.add_argument("--gcn_topk",type=int,default=5)
        self.parser.add_argument("--obj_clip", dest="obj_clip", action="store_true", default=False)
        # A2C
        self.parser.add_argument("--gamma", default=0.9, type=float)
        self.parser.add_argument("--normalize", dest="normalize_loss", default="total", type=str, help='batch or total')

        self.args = self.parser.parse_args()
        args = self.args
        if args.task == 'R2R':
            args.TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
            args.TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'
        elif args.task== 'REVERIE':
            args.TRAIN_VOCAB = 'tasks/REVERIE/REVERIE/train_vocab.txt'
            args.TRAINVAL_VOCAB = 'tasks/REVERIE/REVERIE/train_vocab.txt'
        args.log_dir = 'snap/%s' % args.name
        self.args=args
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        current_path = os.getcwd()
        git_path = os.path.dirname(__file__)
        import git
        repo = git.Repo('./methods/SEvol/')
        commit_hash = repo.head.object.hexsha
        self.args.commit_hash = commit_hash
        import json
        json.dump(vars(self.args),open(os.path.join(args.log_dir,'args_list.json'),'w'))
        if self.args.optim == 'rms':
            print("Optimizer: Using RMSProp")
            self.args.optimizer = torch.optim.RMSprop
        elif self.args.optim == 'adam':
            print("Optimizer: Using Adam")
            self.args.optimizer = torch.optim.Adam
        elif self.args.optim == 'sgd':
            print("Optimizer: sgd")
            self.args.optimizer = torch.optim.SGD
        else:
            assert False

param = Param()
args = param.args
DEBUG_FILE = open(os.path.join('snap', args.name, "debug.log"), 'w')

params = {
    'lr': args.lr,
    'batchSize': args.batchSize,
    'gcn_dim': args.gcn_dim,
    'mlWeight': args.mlWeight
}
#optimized_params = nni.get_next_parameter()
#params.update(optimized_params)
