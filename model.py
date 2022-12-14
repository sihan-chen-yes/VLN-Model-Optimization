
from numpy.core.fromnumeric import clip
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules import activation
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from param import args
from SEM import SEM
import math
from param import params


class ObjEncoder(nn.Module):
    ''' Encodes object labels'''

    def __init__(self, vocab_size, embedding_size, glove_matrix):
        super(ObjEncoder, self).__init__()

        padding_idx = 100
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.embedding.weight.data[...] = torch.from_numpy(glove_matrix)
        self.embedding.weight.requires_grad = False

    def forward(self, inputs):
        embeds = self.embedding(inputs)
        return embeds

class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, dropout_ratio):
        super(EncoderLSTM, self).__init__()
        self.drop = nn.Dropout(p=dropout_ratio)

    def forward(self, word_level_features, sent_level_features):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        ctx = self.drop(word_level_features)
        decoder_init = self.drop(sent_level_features)
        c_t = decoder_init

        return ctx, decoder_init, c_t  # (batch, seq_len, hidden_size)
                                 # (batch, hidden_size)

class SoftDotAttention(nn.Module):
    '''Soft Dot Attention. 

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask.bool(), -float('inf'))
        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, weighted_context, attn
        else:
            return weighted_context, attn

class ScaledSoftDotAttention(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, output_dim):
        super(ScaledSoftDotAttention, self).__init__()
        self.scale = 1 / (output_dim**0.5)
        self.linear_q = nn.Linear(q_dim, output_dim, bias=False)
        self.linear_k = nn.Linear(k_dim, output_dim, bias=False)
        self.linear_v = nn.Sequential(nn.Linear(v_dim, output_dim), nn.Tanh())
    
    def forward(self, q_in, k_in, v_in, mask=None):
        '''
        q = B x L x D
        k = B x L x N x D
        v = B x L x N x D
        mask = B x L x N
        '''
        q = self.linear_q(q_in)
        k = self.linear_k(k_in)
        v = self.linear_v(v_in)
        attn = torch.matmul(k, q.unsqueeze(3)).squeeze(3) * self.scale
        if mask is not None:
            attn.masked_fill_(mask.bool(), -1e9)
        attn = F.softmax(attn, dim=-1)
        v_out = torch.matmul(v.permute(0,1,3,2), attn.unsqueeze(3)).squeeze(3)

        return v_out

class ASODecoderLSTM(nn.Module):
    def __init__(self, action_embed_size, hidden_size, dropout_ratio):
        super(ASODecoderLSTM, self).__init__()
        self.action_embed_size = action_embed_size
        self.hidden_size = hidden_size
        self.action_embedding = nn.Sequential(nn.Linear(args.angle_feat_size, action_embed_size), nn.Tanh())
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)
        self.feat_att_layer = SoftDotAttention(hidden_size, args.visual_feat_size+args.angle_feat_size)
        self.lstm = nn.LSTMCell(action_embed_size+args.visual_feat_size+args.angle_feat_size+params['gcn_dim'], hidden_size)

        self.action_att_layer = SoftDotAttention(hidden_size, hidden_size)
        self.subject_att_layer = SoftDotAttention(hidden_size, hidden_size)
        self.object_att_layer = SoftDotAttention(hidden_size, hidden_size)
        self.graph_att_layer = SoftDotAttention(hidden_size,hidden_size)
        self.topk_att_layer = SoftDotAttention(hidden_size,hidden_size)
        self.fuse_a = nn.Linear(hidden_size, 1)
        self.fuse_s = nn.Linear(hidden_size, 1)
        self.fuse_o = nn.Linear(hidden_size, 1)
        self.static_weights = nn.Parameter(torch.Tensor(params['gcn_dim'],params['gcn_dim']))
        self.reset_param(self.static_weights)
        self.value_action = nn.Sequential(nn.Linear(args.angle_feat_size, hidden_size), nn.Tanh())
        self.subject_att = ScaledSoftDotAttention(args.angle_feat_size, args.angle_feat_size, args.visual_feat_size, hidden_size)    
        self.object_att = ScaledSoftDotAttention(hidden_size, args.clip_dim+args.angle_feat_size, args.clip_dim+args.angle_feat_size,
        hidden_size)
        self.object_graph_att_in = SoftDotAttention(hidden_size,params['gcn_dim'])
        self.object_graph_att = SoftDotAttention(hidden_size,args.out_feats)
        self.object_mapping = nn.Sequential(nn.Linear(args.visual_feat_size+args.angle_feat_size,params['gcn_dim']),nn.Tanh())
        self.object_mapping_out = nn.Sequential(nn.Linear(params['gcn_dim'],args.out_feats),nn.Tanh())
        self.lstm_out_mapping = nn.Sequential(nn.Linear(args.out_feats+hidden_size,hidden_size),nn.Tanh())
        if args.egcn_activation == 'relu':
            self.activation = torch.nn.RReLU()
        self.SEM = SEM(args, self.activation)
#        cand attention layer
        self.cand_att_a = SoftDotAttention(hidden_size, hidden_size)
        self.cand_att_s = SoftDotAttention(hidden_size, hidden_size)
        self.cand_att_o = SoftDotAttention(hidden_size, hidden_size)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def compute_adj_list(self,near_id_feat):
        current_list = near_id_feat.unsqueeze(-1).expand(-1,-1,near_id_feat.shape[1])
        target_list = near_id_feat.unsqueeze(1).expand(-1,near_id_feat.shape[1],-1)
        mod_current = current_list%12
        mod_target = target_list % 12
        distance =  6 - mod_current
        norm_target = (mod_target+distance)%12
        if args.distance_decay_function == 'exp':
            weight = 1/torch.exp(torch.abs(6-norm_target).type(dtype=torch.float32))
        else:
            weight = 1/(torch.abs(6-norm_target).type(dtype=torch.float32)+1)
        # di = weight.sum(dim=1)
        # di = di.pow(-1/2)
        # weight = di.unsqueeze(1)*weight*di.unsqueeze(-1)
        return weight

    def forward(self, action, feature,
                cand_visual_feat, cand_angle_feat, cand_obj_feat,
                near_visual_mask, near_visual_feat, near_angle_feat,
                near_obj_mask, near_obj_feat, near_edge_feat,near_id_feat,
                h_0, prev_h1, c_0,
                ctx, ctx_mask=None,
                already_dropfeat=False):
        '''
        action:batch angle_feat_size

        feature:batch 36 visual_feat_size + angle_feat_size

        candidate: 
        cand_visual_feat:batch cand_len visual_feat_size
        cand_angle_feat:batch cand_len angle_feat_size
        cand_obj_feat???batch cand_len obj_num obj_feat_size

        neighbor:5 neighbor
        near_visual_mask:batch cand_len 5 
        near_visual_feat:batch cand_len 5 visual_feat_size
        near_angle_feat:batch cand_len 5 angle_feat_size
        near_obj_mask:batch cand_len 4
        near_obj_feat:batch cand_len 4 obj_num obj_feat_size
        near_edge_feat:batch cand_len 4 angle_feat_size
        near_id_feat:batch cand_len 5
        '''
        cand_len = cand_obj_feat.shape[1]
        #16 128 -> 16 64  batch angle_feat_size -> batch action_embs_size
        action_embeds = self.action_embedding(action)
        action_embeds = self.drop(action_embeds)
        if not already_dropfeat:
            feature[..., :-args.angle_feat_size] = self.drop_env(feature[..., :-args.angle_feat_size])
            cand_visual_feat = self.drop_env(cand_visual_feat)
            near_visual_feat = self.drop_env(near_visual_feat)
        #16 10 8 640 batch cand_len object_num visual_feat_size
        cand_obj_feat = self.drop_env(cand_obj_feat)
        #16 10 4 8 640 batch cand_len 4 object visual_feat_size
        near_obj_feat = self.drop_env(near_obj_feat)
        #16 10 5 8 640 batch candiate 4 + 1 object visual_feat_size
        object_graph_feat = torch.cat((cand_obj_feat.unsqueeze(2),near_obj_feat),2) 
        #16 10 5 -> 16 10 5 8
        near_id_feat = near_id_feat.unsqueeze(-1).expand(-1,-1,-1,object_graph_feat.shape[3])
        #16 10 5 128 -> 16 10 5 8 128
        angle_graph_feat = near_angle_feat.unsqueeze(3).expand(-1,-1,-1,object_graph_feat.shape[3],-1)
        #16 10 5 8 640 -> 16 400 640 compress
        object_graph_feat = object_graph_feat.reshape(near_obj_feat.shape[0],-1,near_obj_feat.shape[-1])
        #16 10 5 8 128 -> 16 400 128
        angle_graph_feat = angle_graph_feat.reshape(near_angle_feat.shape[0],-1,angle_graph_feat.shape[-1])
        #16 10 5 8 -> 16 400
        near_id_feat = near_id_feat.reshape(near_id_feat.shape[0],-1)
        #16 400 768 batch cand_len * 5 * obj_num visual_feat_size + angle_feat_size assume cand_len * 5 * obj_num == N
        object_graph_feat = torch.cat((object_graph_feat,angle_graph_feat),2)
        if args.distance_decay_function =='same':
            #16 400 400 batch N
            adj_list = torch.ones(object_graph_feat.shape[0],object_graph_feat.shape[1],object_graph_feat.shape[1]).cuda()
        else:
            adj_list = self.compute_adj_list(near_id_feat)
        #TODO
        #16 400
        cand_obj_mask = torch.zeros(object_graph_feat.shape[0],cand_len * 1 * args.top_N_obj).cuda()
        obj_mask = torch.cat((cand_obj_mask, near_obj_mask.unsqueeze(3).expand(-1,-1,-1,args.top_N_obj).contiguous().view(object_graph_feat.shape[0], -1)),1)
        #TODO
        #16 400 768 -> 16 400 128 batch N visual_feat_size + angle_feat_size ->  batch N gcn_dim
        object_graph_feat = self.object_mapping(object_graph_feat)
        #16 640 batch rnn_dim
        prev_h1_drop = self.drop(prev_h1)
        #feature 16 36 768 batch 36 visual_feat_size + angle_feat_size
        #attn_feat 16 768 batch visual_feat_size + angle_feat_size
        attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)
        di = adj_list.sum(dim=1)
        di = di.pow(-1/2)
        #16 400 400 di -> di / N
        weight = di.unsqueeze(1)*adj_list*di.unsqueeze(-1)
        #16 400 400 * 16 400 128 * 16 128 128 -> 16 400 128 batch N N batch N gcn_dim batch gcn_dim gcn_dim -> batch N gcn_dim
        node_embs = self.activation(weight.matmul(object_graph_feat.matmul(self.static_weights)))
        #TODO
        #16 128 batch_size gcn_dim
        node_feat,_ = self.object_graph_att_in(prev_h1,node_embs,output_tilde=False)
        #16 64 + 640 + 128 + 128 batch action_embs_size + visual_feat_size + angle_feat_size + gcn_dim
        concat_input = torch.cat((action_embeds, attn_feat, node_feat), dim=-1)
        #16 640 batch rnn_dim
        h_1, c_1 = self.lstm(concat_input, (prev_h1, c_0))
        h_1_drop = self.drop(h_1)
        #16 640 batch rnn_dim
        object_h1_drop = h_1_drop
        #16 77 640 batch L rnn_dim
        object_ctx = ctx.detach()
        #16 640 batch rnn_dim
        selector,_, _ = self.topk_att_layer(object_h1_drop,object_ctx,ctx_mask)
        #SEM
        #16 400 400 batch N N adj_list
        #16 400 128 batch N gcn_dim object_graph_feat 
        #16 77 640 batch L rnn_dim ctx
        #16 400 batch N
        #16 640 batch rnn_dim
        node_feats,policy_score,scorer,entropy_object = self.SEM(adj_list,object_graph_feat,ctx,obj_mask,selector)
        #16 128 128 -> 16 128 300 batch gcn_dim gcn_dim -> batch gcn_dim out_feats
        node_feats = self.object_mapping_out(node_feats)
        #16 300 batch out_feats
        node_feat, _ = self.object_graph_att(h_1_drop,node_feats, output_tilde=False)
        #16 640 + 300 -> 16 640  batch rnn_dim + out_feats -> batch rnn_dim
        h_1_drop = self.drop(self.lstm_out_mapping(torch.cat([h_1_drop,node_feat],-1)))
        h_a, u_a, _ = self.action_att_layer(h_1_drop, ctx, ctx_mask)
        h_s, u_s, _ = self.subject_att_layer(h_1_drop, ctx, ctx_mask)
        h_o, u_o, _ = self.object_att_layer(h_1_drop, ctx, ctx_mask)

        h_a_drop, u_a_drop = self.drop(h_a), self.drop(u_a)
        h_s_drop, u_s_drop = self.drop(h_s), self.drop(u_s)
        h_o_drop, u_o_drop = self.drop(h_o), self.drop(u_o)
        fusion_weight = torch.cat([self.fuse_a(u_a_drop), self.fuse_s(u_s_drop), self.fuse_o(u_o_drop)], dim=-1)
        fusion_weight = F.softmax(fusion_weight, dim=-1)
        #batch candidate
        B, L = near_visual_mask.shape[0], near_visual_mask.shape[1]
        #action
        # B L hidden_size
        v_action = self.value_action(cand_angle_feat)

        #subject
        v_subject = self.subject_att(cand_angle_feat, near_angle_feat, near_visual_feat, near_visual_mask)
        v_subject = self.drop(v_subject)

        #object
        near_obj = torch.cat([near_obj_feat, near_edge_feat.unsqueeze(3).expand(-1,-1,-1,args.top_N_obj,-1)], dim=-1)
        near_obj = near_obj.view(B, L, 4*args.top_N_obj, -1)
        near_obj_mask = near_obj_mask.unsqueeze(3).expand(-1,-1,-1,args.top_N_obj).contiguous().view(B, L, 4*args.top_N_obj)
        v_object = self.object_att(u_o_drop.unsqueeze(1).expand(-1,L,-1), near_obj, near_obj, near_obj_mask)
        v_object = self.drop(v_object)

        _, logit_a = self.cand_att_a(h_a_drop, v_action, output_tilde=False, output_prob=False)
        _, logit_s = self.cand_att_s(h_s_drop, v_subject, output_tilde=False, output_prob=False)
        _, logit_o = self.cand_att_o(h_o_drop, v_object, output_tilde=False, output_prob=False)
        logit = torch.cat([logit_a.unsqueeze(2), logit_s.unsqueeze(2), logit_o.unsqueeze(2)], dim=-1)
        logit = torch.matmul(logit, fusion_weight.unsqueeze(2)).squeeze(2)
        h_tilde = (h_a + h_s + h_o) / 3.

        return h_1, c_1, logit, h_tilde,policy_score,scorer,entropy_object

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(args.rnn_dim, args.rnn_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.rnn_dim, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()

class Critic_object(nn.Module):
    def __init__(self):
        super(Critic_object, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(params['gcn_dim'], params['gcn_dim']),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(params['gcn_dim'], 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()


class SpeakerEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout_ratio, bidirectional):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.feature_size = feature_size

        if bidirectional:
            print("BIDIR in speaker encoder!!")

        self.lstm = nn.LSTM(feature_size, self.hidden_size // self.num_directions, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop3 = nn.Dropout(p=args.featdropout)
        self.attention_layer = SoftDotAttention(self.hidden_size, feature_size)

        self.post_lstm = nn.LSTM(self.hidden_size, self.hidden_size // self.num_directions, self.num_layers,
                                 batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)

    def forward(self, action_embeds, feature, lengths, already_dropfeat=False):
        """
        :param action_embeds: (batch_size, length, 2048 + 128). The feature of the view
        :param feature: (batch_size, length, 36, 2048 + 128). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        x = action_embeds
        if not already_dropfeat:
            x[..., :-args.angle_feat_size] = self.drop3(x[..., :-args.angle_feat_size])            # Do not dropout the spatial features

        # LSTM on the action embed
        ctx, _ = self.lstm(x)
        ctx = self.drop(ctx)

        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        if not already_dropfeat:
            feature[..., :-args.angle_feat_size] = self.drop3(feature[..., :-args.angle_feat_size])   # Dropout the image feature
        x, _, _ = self.attention_layer(                        # Attend to the feature map
            ctx.contiguous().view(-1, self.hidden_size),    # (batch, length, hidden) --> (batch x length, hidden)
            feature.view(batch_size * max_length, -1, self.feature_size),        # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
        )
        x = x.view(batch_size, max_length, -1)
        x = self.drop(x)

        # Post LSTM layer
        x, _ = self.post_lstm(x)
        x = self.drop(x)

        return x

class SpeakerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout_ratio)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)
        self.baseline_projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(128, 1)
        )

    def forward(self, words, ctx, ctx_mask, h0, c0):
        embeds = self.embedding(words)
        embeds = self.drop(embeds)
        x, (h1, c1) = self.lstm(embeds, (h0, c0))

        x = self.drop(x)

        # Get the size
        batchXlength = words.size(0) * words.size(1)
        multiplier = batchXlength // ctx.size(0)         # By using this, it also supports the beam-search

        # Att and Handle with the shape
        # Reshaping x          <the output> --> (b(word)*l(word), r)
        # Expand the ctx from  (b, a, r)    --> (b(word)*l(word), a, r)
        # Expand the ctx_mask  (b, a)       --> (b(word)*l(word), a)
        x, _, _ = self.attention_layer(
            x.contiguous().view(batchXlength, self.hidden_size),
            ctx.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous(). view(batchXlength, -1, self.hidden_size),
            mask=ctx_mask.unsqueeze(1).expand(-1, multiplier, -1).contiguous().view(batchXlength, -1)
        )
        x = x.view(words.size(0), words.size(1), self.hidden_size)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        return logit, h1, c1
    
if args.CLIP_language:
    from myclip import clip
    class CLIP_language(nn.Module):

        def __init__(self):
            super().__init__()
            self.model,preprocess = clip.load('RN50x4')
        
        def forward(self,text_list):
            with torch.no_grad():
                text = clip.tokenize(text_list,truncate=True).cuda()
                word_level_features,sent_level_features,mask= self.model.encode_text(text)
            return word_level_features,sent_level_features,mask
