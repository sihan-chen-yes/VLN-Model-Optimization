import utils as u
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
from param import args

class EGCN(torch.nn.Module):
    def __init__(self, args, activation, skipfeats=False):
        super().__init__()
        GRCU_args = u.Namespace({})

        feats = [args.feats_per_node,
                 args.layer_1_feats,
                 args.layer_2_feats]
        self.skipfeats = skipfeats
        self.GRCU_layers = []
        self._parameters = nn.ParameterList()
        for i in range(1,len(feats)):
            GRCU_args = u.Namespace({'in_feats' : feats[i-1],
                                     'out_feats': feats[i],
                                     'activation': activation})

            grcu_i = GRCU(GRCU_args)
            self.__setattr__(f'grcu_{i}',grcu_i)
            #print (i,'grcu_i', grcu_i)
            #self.GRCU_layers.append(grcu_i.to(self.device))
            self.GRCU_layers.append(grcu_i)
            self._parameters.extend(list(self.GRCU_layers[-1].parameters()))

    def parameters(self):
        return self._parameters

    def forward(self,A_list, Nodes_list,nodes_mask_list):
        node_feats= Nodes_list[-1]

        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list,Nodes_list,nodes_mask_list)

        out = Nodes_list[-1]
        if self.skipfeats:
            out = torch.cat((out,node_feats), dim=1)   # use node_feats.to_dense() if 2hot encoded input 
        return out


class GRCU(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        cell_args = u.Namespace({})
        cell_args.rows = args.in_feats
        cell_args.cols = args.out_feats

        self.evolve_weights = mat_GRU_cell(cell_args)

        self.activation = self.args.activation
        self.GCN_init_weights = Parameter(torch.Tensor(self.args.in_feats,self.args.out_feats))
        self.reset_param(self.GCN_init_weights)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,A_list,node_embs_list,mask_list):
        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t,Ahat in enumerate(A_list):
            node_embs = node_embs_list[t]
            #first evolve the weights from the initial and use the new weights with the node_embs
            GCN_weights = self.evolve_weights(GCN_weights,node_embs,mask_list[t])
            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))

            out_seq.append(node_embs)

        return out_seq


class GRCU_Cell(torch.nn.Module):
    def __init__(self,args,activation):
        super().__init__()
        self.args = args
        cell_args = u.Namespace({})
        cell_args.rows = args.gcn_dim
        cell_args.cols = args.gcn_dim

        self.evolve_weights = mat_GRU_cell(cell_args)

        self.activation = activation
        self.GCN_init_mapping = Parameter(torch.Tensor(1,args.gcn_dim))
        self.init_mapping = nn.Sequential(nn.Linear(args.rnn_dim,args.gcn_dim),nn.Tanh())
        if args.static_gcn_weights or args.static_gcn_weights_only:
            self.static_weights = nn.Parameter(torch.Tensor(args.gcn_dim,args.gcn_dim))
            self.reset_param(self.static_weights)
        self.reset_param(self.GCN_init_mapping)
        self.GCN_pre_weights = None
        self.GCN_init_weights = None
    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def init_weights(self,ht):
        ht = self.init_mapping(ht)
        self.GCN_pre_weights = torch.matmul(ht.unsqueeze(-1),self.GCN_init_mapping)
        self.GCN_init_weights = self.GCN_pre_weights

    def forward(self,Ahat,node_embs,mask,ht=None):
            #first evolve the weights from the initial and use the new weights with the node_embs
        policy_score = None
        if self.GCN_pre_weights is not None:
            GCN_weights,policy_score = self.evolve_weights(self.GCN_pre_weights,node_embs,mask,ht)
            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))
            if args.static_gcn_weights:
                node_embs1 = self.activation(Ahat.matmul(node_embs.matmul(self.static_weights)))
                node_embs = (node_embs+node_embs1)/2
            if args.static_gcn_weights_only:
                node_embs1 = self.activation(Ahat.matmul(node_embs.matmul(self.static_weights)))
                node_embs = node_embs1
            self.GCN_pre_weights = GCN_weights
        return node_embs,policy_score        

class mat_GRU_cell(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.update = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Tanh())
        
        # self.choose_topk = TopK(feats = args.rows,
        #                         k = args.cols)
        self.choose_topk = TopK_with_h()

    def forward(self,prev_Q,prev_Z,mask,ht):


        z_topk,policy_score = self.choose_topk(prev_Z,mask,ht)
        update = self.update(z_topk,prev_Q)
        reset = self.reset(z_topk,prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q,policy_score

        

class mat_GRU_gate(torch.nn.Module):
    def __init__(self,rows,cols,activation):
        super().__init__()
        self.activation = activation
        #the k here should be in_feats which is actually the rows
        self.W = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows,cols))

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,x,hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out

class TopK(torch.nn.Module):
    def __init__(self,feats,k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats,1))
        self.reset_param(self.scorer)
        
        self.k = k

    def reset_param(self,t):
        #Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv,stdv)

    def forward(self,node_embs,mask):
        batch_size,graph_size,feat_size = node_embs.shape
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        scores = scores.squeeze() + mask
        vals, topk_indices = scores.view(batch_size,-1).topk(self.k,dim=1)
        topk_indices = topk_indices[vals > -float("Inf")].view(batch_size,-1)
        if topk_indices.size(1) < self.k:
            topk_indices = u.pad_with_last_val(topk_indices,self.k)
        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
           isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()
        out = node_embs.gather(1,topk_indices.unsqueeze(-1).expand(-1,-1,feat_size)) * tanh(scores.gather(1,topk_indices)).unsqueeze(-1)
        #we need to transpose the output
        return out.transpose(1,2)
class TopK_with_h(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mapper =  nn.Sequential(nn.Linear(args.rnn_dim,args.gcn_dim),nn.Tanh())
        self.softmax = nn.Softmax()
        self.k = args.gcn_dim
    def reset_param(self,t):
        return None
    def forward(self,node_embs,mask,h_t = None):
        batch_size,graph_size,feat_size = node_embs.shape
        scorer = self.mapper(h_t)
        scores = node_embs.bmm(scorer.unsqueeze(-1)).squeeze()/scorer.norm(dim=1).unsqueeze(-1)
        scores = scores.squeeze() + mask
        vals, topk_indices = scores.view(batch_size,-1).topk(self.k,dim=1)
        topk_indices = topk_indices[vals > -float("Inf")].view(batch_size,-1)
        if topk_indices.size(1) < self.k:
            topk_indices = u.pad_with_last_val(topk_indices,self.k)
        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
           isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()
        out = node_embs.gather(1,topk_indices.unsqueeze(-1).expand(-1,-1,feat_size)) * tanh(scores.gather(1,topk_indices)).unsqueeze(-1)
        scores = self.softmax(scores.view(batch_size,-1))
        c = scores.log()
        score =  c.gather(-1,topk_indices)
        policy_score = score.mean(dim=1)
        #we need to transpose the output
        return out.transpose(1,2),policy_score