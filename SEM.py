import utils as u
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
from param import args
from param import params

class SEM(torch.nn.Module):
    def __init__(self,args,activation):
        super().__init__()
        self.args = args
        cell_args = u.Namespace({})
        cell_args.rows = args.gcn_dim
        cell_args.cols = args.gcn_dim
        self.mapper = nn.Linear(args.rnn_dim,params['gcn_dim'])
        self.evolve_weights = mat_GRU_cell(cell_args)
        self.activation = activation
        self.GCN_init_mapping = Parameter(torch.Tensor(1,params['gcn_dim']))
        self.init_mapping = nn.Sequential(nn.Linear(args.rnn_dim,params['gcn_dim']),nn.Tanh())
        self.static_weights = nn.Parameter(torch.Tensor(params['gcn_dim'],params['gcn_dim']))
        self.reset_param(self.static_weights)
        self.reset_param(self.GCN_init_mapping)
        self.GCN_weights = None
        # self.evolve_A = nn.Parameter(torch.Tensor(params['batchSize'],args.gcn_topk,args.gcn_topk))
        # self.reset_param(self.evolve_A)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def init_weights(self,ht):
        #batch rnn_dim -> batch gcn_dim gcn_dim
        ht = self.init_mapping(ht)
        self.GCN_weights = torch.matmul(ht.unsqueeze(-1),self.GCN_init_mapping)

    def forward(self,Ahat,node_embs,word_level_features,mask,ht=None):
        # #first evolve the weights from the initial and use the new weights with the node_embs

        # #batch gcn_topk gcn_topk
        # self.evolve_A.data,policy_score,scorer,entropy_object,selected_indexes = self.evolve_weights(self.evolve_A,node_embs,mask,ht)

        # #batch len feat_size
        # word_level_features = self.mapper(word_level_features)
        # attn = node_embs.matmul(word_level_features.permute(0,2,1))
        # #batch N N -> batch gcn_topk gcn_topk
        # attn_A = attn.matmul(attn.permute(0,2,1))
        # attn_A = attn_A.gather(1,selected_indexes.unsqueeze(-1).expand(-1,-1,attn_A.shape[-1]))
        # attn_A = attn_A.gather(2,selected_indexes.unsqueeze(1).expand(-1,attn_A.shape[1],-1))

        # #batch N N -> batch gcn_topk gcn_topk
        # Ahat = Ahat.gather(1,selected_indexes.unsqueeze(-1).expand(-1,-1,Ahat.shape[-1]))
        # Ahat = Ahat.gather(2,selected_indexes.unsqueeze(1).expand(-1,Ahat.shape[1],-1))
        # di = Ahat.sum(dim=1)
        # di = di.pow(-1/2)
        # Ahat = di.unsqueeze(1)*Ahat*di.unsqueeze(-1).detach()
        # #batch gcn_topk gcn_topk
        # self.evolve_A.data = self.evolve_A + Ahat + attn_A
        # node_embs = node_embs.gather(1,selected_indexes.unsqueeze(-1).expand(-1,-1,node_embs.shape[-1]))
        # node_embs = self.activation(self.evolve_A.matmul(node_embs.matmul(self.GCN_weights)))

        # if args.static_gcn_weights:
        #     node_embs1 = self.activation(self.evolve_A.matmul(node_embs.matmul(self.static_weights)))
        #     node_embs = (node_embs+node_embs1)/2
        # if args.static_gcn_weights_only:
        #     node_embs1 = self.activation(self.evolve_A.matmul(node_embs.matmul(self.static_weights)))
        #     node_embs = node_embs1
        # return node_embs,policy_score,scorer,entropy_object

        self.GCN_weights,policy_score,scorer,entropy_object,selected_indexes = self.evolve_weights(self.GCN_weights,node_embs,mask,ht)
        node_embs = node_embs.gather(1,selected_indexes.unsqueeze(-1).expand(-1,-1,node_embs.shape[-1]))
        Ahat = Ahat.gather(1,selected_indexes.unsqueeze(-1).expand(-1,-1,Ahat.shape[-1]))
        Ahat = Ahat.gather(2,selected_indexes.unsqueeze(1).expand(-1,Ahat.shape[1],-1))
        di = Ahat.sum(dim=1)
        di = di.pow(-1/2)
        Ahat =  di.unsqueeze(1)*Ahat*di.unsqueeze(-1).detach()
        node_embs = self.activation(Ahat.matmul(node_embs.matmul(self.GCN_weights)))
        node_embs1 = self.activation(Ahat.matmul(node_embs.matmul(self.static_weights)))
        node_embs = (node_embs+node_embs1)/2
        return node_embs,policy_score,scorer,entropy_object

class mat_GRU_cell(torch.nn.Module):
    def __init__(self,args1):
        super().__init__()
        self.args = args1
        self.update = mat_GRU_gate(args1.rows,
                                   args1.cols,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(args1.rows,
                                   args1.cols,
                                   torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(args1.rows,
                                   args1.cols,
                                   torch.nn.Tanh())
        
        # self.choose_topk = TopK(feats = args.rows,
        #                         k = args.cols)
        self.choose_topk = TopK_with_h()

    def forward(self,prev_D,node_embs,mask,ht):
        # evolve_A:Q 
        # node_emb:Z
        #select first then evolve
        #selected_node_embs:batch gcn_dim gcn_dim
        selected_node_embs,policy_score,scorer,entropy_object,topk_indices = self.choose_topk(node_embs,mask,ht)
        update = self.update(selected_node_embs,prev_D)
        reset = self.reset(selected_node_embs,prev_D)

        prev_D_tilde = reset * prev_D
        prev_D_tilde = self.htilda(selected_node_embs, prev_D_tilde)

        D = (1 - update) * prev_D + update * prev_D_tilde

        return D,policy_score,scorer,entropy_object,topk_indices

        

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

# class TopK(torch.nn.Module):
#     def __init__(self,feats,k):
#         super().__init__()
#         self.scorer = Parameter(torch.Tensor(feats,1))
#         self.reset_param(self.scorer)
        
#         self.k = k

#     def reset_param(self,t):
#         #Initialize based on the number of rows
#         stdv = 1. / math.sqrt(t.size(0))
#         t.data.uniform_(-stdv,stdv)

#     def forward(self,node_embs,mask):
#         batch_size,graph_size,feat_size = node_embs.shape
#         scores = node_embs.matmul(self.scorer) / self.scorer.norm()
#         scores = scores.squeeze() + mask
#         vals, topk_indices = scores.view(batch_size,-1).topk(self.k,dim=1)
#         topk_indices = topk_indices[vals > -float("Inf")].view(batch_size,-1)
#         if topk_indices.size(1) < self.k:
#             topk_indices = u.pad_with_last_val(topk_indices,self.k)
#         tanh = torch.nn.Tanh()

#         if isinstance(node_embs, torch.sparse.FloatTensor) or \
#            isinstance(node_embs, torch.cuda.sparse.FloatTensor):
#             node_embs = node_embs.to_dense()
#         out = node_embs.gather(1,topk_indices.unsqueeze(-1).expand(-1,-1,feat_size)) * tanh(scores.gather(1,topk_indices)).unsqueeze(-1)
#         #we need to transpose the output
#         return out.transpose(1,2)

class TopK_with_h(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #TODO change dim
        self.mapper = nn.Sequential(nn.Linear(args.rnn_dim,params['gcn_dim']),nn.Tanh())
        self.softmax = nn.Softmax()

    def reset_param(self,t):
        return None

    def forward(self,node_embs,mask,h_selector = None):
        """
        node_embs:64 400 128 batch_size node_num gcn_dim
        h_t:64 640 batch rnn_dim
        """
        batch_size,graph_size,feat_size = node_embs.shape
        #16 128 batch gcn_dim
        scorer = self.mapper(h_selector.detach())
        #16 400 128 * 16 128 1 -> 16 400 batch N 
        scores = node_embs.bmm(scorer.unsqueeze(-1)).squeeze()
        #16 400 batch N   scores of obj
        scores = scores / (scorer.norm(dim=1).unsqueeze(-1))
        # scores = scores.squeeze()
        # select top K obj
        #16 5 batch top_K
        vals, topk_indices = scores.view(batch_size,-1).topk(args.gcn_topk,dim=1)
        # topk_indices_out = topk_indices[vals > -float("Inf")].view(batch_size,-1)
        # if topk_indices.size(1) < self.k:
        #     topk_indices = u.pad_with_last_val(topk_indices,self.k)
        #to match GRUm size
        repeat_num = args.gcn_dim / args.gcn_topk + 1
        # batch_size gcn_dim
        topk_indices = topk_indices.repeat(1,int(repeat_num))[:,:args.gcn_dim]

        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
           isinstance(node_embs, torch.cuda.sparse.FloatTensor):
           #for sparse matrix
            node_embs = node_embs.to_dense()
        topK_node_embs = node_embs.gather(1,topk_indices.unsqueeze(-1).expand(-1,-1,feat_size))
        #use scores to highlight
        topK_node_embs = topK_node_embs * tanh(scores.gather(1,topk_indices)).unsqueeze(-1)
        scores = self.softmax(scores.view(batch_size,-1))
        entropy_object = torch.distributions.Categorical(scores).entropy()
        c = scores.log()
        topK_scores =  c.gather(-1,topk_indices)
        score_policy = topK_scores.mean(dim=1)
        #we need to transpose the output so that GRUm works
        return topK_node_embs.transpose(1,2),score_policy,scorer,entropy_object,topk_indices
