
import json
import os
import sys
from ipdb.__main__ import set_trace
import numpy as np
import random
import math
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from env import R2RBatch
from utils import padding_idx, add_idx, Tokenizer
import utils
import model
import param
from param import args
from collections import defaultdict
#import nni
from param import params

class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents
    
    def write_results(self):
        output = [{'instr_id':k, 'trajectory': v} for k,v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def get_results(self):
        output = [{'instr_id': k, 'trajectory': v} for k, v in self.results.items()]
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj['path']
        else:   # Do a full round
            while True:
                for traj in self.rollout(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj['path']
                if looped:
                    break

class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    env_actions = {
      'left': (0,-1, 0), # left
      'right': (0, 1, 0), # right
      'up': (0, 0, 1), # up
      'down': (0, 0,-1), # down
      'forward': (1, 0, 0), # forward
      '<end>': (0, 0, 0), # <end>
      '<start>': (0, 0, 0), # <start>
      '<ignore>': (0, 0, 0)  # <ignore>
    }

    def __init__(self, env, results_path, tok, episode_len=20):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.tok = tok
        self.episode_len = episode_len
        self.feature_size = self.env.feature_size

        if args.CLIP_language:
            self.CLIP_language = model.CLIP_language()
        # objencoder
        with open('img_features/objects/object_vocab.txt', 'r') as f_ov:
            obj_vocab = [k.strip() for k in f_ov.readlines()]
        if args.obj_clip:
            obj_matrix = utils.get_clip_matrix(obj_vocab, args.clip_dim)
        else:
            obj_matrix = utils.get_glove_matrix(obj_vocab, args.glove_dim)
        # Models
        self.objencoder = model.ObjEncoder(obj_matrix.shape[0], obj_matrix.shape[1], obj_matrix).cuda()
        self.encoder = model.EncoderLSTM(args.dropout).cuda()
        self.decoder = model.ASODecoderLSTM(args.aemb, args.rnn_dim, args.dropout).cuda()
        self.critic = model.Critic().cuda()
        #TODO
        self.models = (self.decoder, self.critic)
        #TODO
        self.critic_object = model.Critic_object().cuda()


        # Optimizers
        self.decoder_optimizer = args.optimizer(self.decoder.parameters(), lr=params['lr'])
        self.critic_optimizer = args.optimizer(self.critic.parameters(), lr=params['lr'])
        #TODO
        self.optimizers = (self.decoder_optimizer, self.critic_optimizer)
        #TODO critic_object_optimzer bug here!
        self.critic_object_optimizer = args.optimizer(self.critic_object.parameters(), lr=params['lr'])

        # Evaluations
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, size_average=False)
        self.ndtw_criterion = utils.ndtw_initialize()

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)


    def _sort_batch(self, obs):
        ''' Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). '''
        #ob for each agent 
        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]     # Full length

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)       # True -> descending
        sorted_tensor = seq_tensor[perm_idx]
        mask = (sorted_tensor == padding_idx)[:,:seq_lengths[0]]    # seq_lengths[0] is the Maximum length

        return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
               mask.byte().cuda(),  \
               list(seq_lengths), list(perm_idx)

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        #get visual feat
        #views == 36 == 3 * 12
        features = np.empty((len(obs), args.views, args.visual_feat_size + args.angle_feat_size), dtype=np.float32)
        for i, ob in enumerate(obs):
            features[i, :, :] = ob['feature']   # Image feat
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def _candidate_variable(self, obs):
        cand_leng = [len(ob['candidate']) + 1 for ob in obs]       # +1 is for the end
        #get greatest number of possible views and + 1 for end op
        max_cand_leng = max(cand_leng)

        cand_visual_feat = np.zeros((len(obs), max_cand_leng, args.visual_feat_size), dtype=np.float32)
        cand_angle_feat = np.zeros((len(obs), max_cand_leng, args.angle_feat_size), dtype=np.float32)
        cand_obj_class = np.zeros((len(obs), max_cand_leng, args.top_N_obj), dtype=np.float32)
        near_visual_mask = np.ones((len(obs), max_cand_leng, 5), dtype=np.uint8)
        near_visual_feat = np.zeros((len(obs), max_cand_leng, 5, args.visual_feat_size), dtype=np.float32)
        near_angle_feat = np.zeros((len(obs), max_cand_leng, 5, args.angle_feat_size), dtype=np.float32)

        near_obj_mask = np.ones((len(obs), max_cand_leng, 4), dtype=np.uint8)
        near_obj_class = np.zeros((len(obs), max_cand_leng, 4, args.top_N_obj), dtype=np.float32)
        near_edge_feat = np.zeros((len(obs), max_cand_leng, 4, args.angle_feat_size), dtype=np.float32)
        near_id_feat = np.zeros((len(obs), max_cand_leng, 5),dtype=np.float32)
        for i, ob in enumerate(obs):
            for j, c in enumerate(ob['candidate']):
                cand_visual_feat[i, j, :] = c['visual_feat']
                cand_angle_feat[i, j, :] = c['angle_feat']
                cand_obj_class[i, j, :] = c['obj_class']
                near_visual_mask[i,j,...] = c['near_mask']
                near_visual_feat[i,j,...] = c['near_visual_feat']
                near_angle_feat[i,j,...] = c['near_angle_feat']
                near_obj_mask[i,j,...] = c['near_mask'][1:,...]
                near_obj_class[i,j,...] = c['near_obj_class'][1:,...]
                near_edge_feat[i,j,...] = c['near_edge_feat'][1:,...]
                near_id_feat[i,j,...] = c['near_view_id']

        cand_visual_feat = torch.from_numpy(cand_visual_feat).cuda()
        cand_angle_feat = torch.from_numpy(cand_angle_feat).cuda()
        cand_obj_feat = self.objencoder(torch.from_numpy(cand_obj_class).cuda().long())

        near_visual_mask = torch.from_numpy(near_visual_mask).cuda()
        near_visual_feat = torch.from_numpy(near_visual_feat).cuda()
        near_angle_feat = torch.from_numpy(near_angle_feat).cuda()

        near_obj_mask = torch.from_numpy(near_obj_mask).cuda()
        near_obj_feat = self.objencoder(torch.from_numpy(near_obj_class).cuda().long())
        near_edge_feat = torch.from_numpy(near_edge_feat).cuda()
        near_id_feat = torch.from_numpy(near_id_feat).cuda()
        return cand_leng, cand_visual_feat, cand_angle_feat, cand_obj_feat, \
               near_visual_mask, near_visual_feat, near_angle_feat, \
               near_obj_mask, near_obj_feat, near_edge_feat, near_id_feat

    def get_input_feat(self, obs):
        input_a_t = None
        f_t = self._feature_variable(obs)      # Image features from obs

        cand_leng, cand_visual_feat, cand_angle_feat, cand_obj_feat, \
        near_visual_mask, near_visual_feat, near_angle_feat, \
        near_obj_mask, near_obj_feat, near_edge_feat, near_id_feat = self._candidate_variable(obs)

        return input_a_t, f_t, \
               cand_leng, cand_visual_feat, cand_angle_feat, cand_obj_feat, \
               near_visual_mask, near_visual_feat, near_angle_feat, \
               near_obj_mask, near_obj_feat, near_edge_feat, near_id_feat
    
    def _teacher_action(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda()

    # def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
    #     """
    #     Interface between Panoramic view and Egocentric view 
    #     It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
    #     """
    #     if perm_idx is None:
    #         perm_idx = range(len(perm_obs))
    #     actions = [[]] * self.env.batch_size # batch * action_len
    #     max_len = 0 # for padding stop action
    #     for i, idx in enumerate(perm_idx):
    #         action = a_t[i]
    #         if action != -1:            # -1 is the <stop> action
    #             select_candidate = perm_obs[i]['candidate'][action]
    #             src_point = perm_obs[i]['viewIndex']
    #             trg_point = select_candidate['pointId']
    #             src_level = (src_point) // 12   # The point idx started from 0
    #             trg_level = (trg_point) // 12
    #             src_heading = (src_point) % 12
    #             trg_heading = (trg_point) % 12
    #             # adjust elevation
    #             if trg_level > src_level:
    #                 actions[idx] = actions[idx] + [self.env_actions['up']] * int(trg_level - src_level)
    #             elif trg_level < src_level:
    #                 actions[idx] = actions[idx] + [self.env_actions['down']] * int(src_level - trg_level)
    #             # adjust heading
    #             if trg_heading > src_heading:
    #                 dif = trg_heading - src_heading
    #                 if dif >= 6: # turn left
    #                     actions[idx] = actions[idx] + [self.env_actions['left']] * int(12 - dif)
    #                 else: # turn right
    #                     actions[idx] = actions[idx] + [self.env_actions['right']] * int(dif)
    #             elif trg_heading < src_heading:
    #                 dif = src_heading - trg_heading
    #                 if dif >=6: # turn right
    #                     actions[idx] = actions[idx] + [self.env_actions['right']] * int(12 - dif)
    #                 else: # turn left
    #                     actions[idx] = actions[idx] + [self.env_actions['left']] * int(dif)

    #             actions[idx] = actions[idx] + [(select_candidate['idx'], 0, 0)]
    #             max_len = max(max_len, len(actions[idx]))

    #     for idx in perm_idx:
    #         if len(actions[idx]) < max_len:
    #             actions[idx] = actions[idx] + [self.env_actions['<end>']] * (max_len - len(actions[idx]))
    #     actions = np.array(actions, dtype = 'float32')

    #     for i in range(max_len):
    #         cur_actions = actions[:,i]
    #         cur_actions = list(cur_actions)
    #         cur_actions = [tuple(a) for a in cur_actions]
    #         self.env.env.makeActions(cur_actions)
        
    #     if traj is not None:
    #         state = self.env.env.sim.getState()
    #         for i, idx in enumerate(perm_idx):
    #             action = a_t[i]
    #             if action != -1:
    #                 traj[i]['path'].append((state[idx].location.viewpointId, state[idx].heading, state[idx].elevation))

    def make_equiv_action(self, a_t, obs, perm_idx, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        new_action_current = a_t.copy()
        if perm_idx is not None:
            for i,idx in enumerate(perm_idx):
                new_action_current[idx] = a_t[i]
        try:
            self.env.env.makeActions(new_action_current+1)
        except:
            import ipdb;ipdb.set_trace()
        for i,ob in enumerate(obs):
            state = self.env.env.sims[perm_idx[i]]
            #traj:batch
            if traj is not None:
                traj[i]['path'].append((state.viewpointId, state.heading, state.elevation))

    def rollout(self, train_ml=None, train_rl=True, reset=True, speaker=None):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment
        :param speaker:     Speaker used in back translation.
                            If the speaker is not None, use back translation.
                            O.w., normal training
        :return:
        """
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset:
            # Reset env
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        batch_size = len(obs)

        if speaker is not None:         # Trigger the self_train mode!
            noise = self.decoder.drop_env(torch.ones(self.feature_size).cuda())
            batch = self.env.batch.copy()
            speaker.env = self.env
            insts = speaker.infer_batch(featdropmask=noise)     # Use the same drop mask in speaker

            # Create fake environments with the generated instruction
            boss = np.ones((batch_size, 1), np.int64) * self.tok.word_to_index['<BOS>']  # First word is <BOS>
            insts = np.concatenate((boss, insts), 1)
            for i, (datum, inst) in enumerate(zip(batch, insts)):
                if inst[-1] != self.tok.word_to_index['<PAD>']: # The inst is not ended!
                    inst[-1] = self.tok.word_to_index['<EOS>']
                datum.pop('instructions')
                datum.pop('instr_encoding')
                datum['instructions'] = self.tok.decode_sentence(inst)
                datum['instr_encoding'] = inst
            obs = np.array(self.env.reset(batch))

        # Reorder the language input for the encoder (do not ruin the original code)
        _, _, _, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]
        if args.CLIP_language:
            row_text= [x['instructions'] for x in perm_obs]
            word_level_features,sent_level_features, ctx_mask = self.CLIP_language(row_text)
            word_level_features = word_level_features.float()
            sent_level_features = sent_level_features.float()
        ctx, h_t, c_t = self.encoder(word_level_features,sent_level_features)
        #TODO need mapper here!
        self.decoder.SEM.init_weights(h_t.detach())
        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in perm_obs]

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        last_ndtw = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(perm_obs):   # The init distance from the view point to the target
            last_dist[i] = ob['distance']
            path_act = [vp[0] for vp in traj[i]['path']]
            if not args.train == "validlistener":
                last_ndtw[i] = self.ndtw_criterion[ob['scan']](path_act, ob['gt_path'], metric='ndtw')

        # Initialization the tracking state
        ended = np.array([False] * batch_size)  # Indices match permuation of the model, not env

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        policy_score_probs = []
        masks = []
        entropys = []
        ml_loss = 0.
        scorers = []
        entropys_object = []
        h1 = h_t
        input_a_t = torch.zeros([len(obs), args.angle_feat_size]).cuda()
        for t in range(self.episode_len):
            #this step
            _, f_t, \
            cand_leng, cand_visual_feat, cand_angle_feat, cand_obj_feat, \
            near_visual_mask, near_visual_feat, near_angle_feat, \
            near_obj_mask, near_obj_feat, near_edge_feat, near_id_feat = self.get_input_feat(perm_obs)
            if speaker is not None:       # Apply the env drop mask to the feat
                f_t[..., :-args.angle_feat_size] *= noise
                cand_visual_feat *= noise
                near_visual_feat *= noise
            h_t, c_t, logit, h1,policy_score,scorer,entropy_object = self.decoder(input_a_t, f_t, 
                                               cand_visual_feat, cand_angle_feat, cand_obj_feat,
                                               near_visual_mask, near_visual_feat, near_angle_feat,
                                               near_obj_mask, near_obj_feat, near_edge_feat, near_id_feat,
                                               h_t, h1, c_t,
                                               ctx, ctx_mask,
                                               already_dropfeat=(speaker is not None))
            #policy_score:mean of topK_scores  batch 
            policy_score_probs.append(policy_score)
            #h_t:batch rnn_dim
            hidden_states.append(h_t)
            #scorer:batch gcn_dim
            scorers.append(scorer)
            #entropy_object:batch
            entropys_object.append(entropy_object)
            # Mask outputs where agent can't move forward
            # Here the logit is [b, max_candidate]
            candidate_mask = utils.length2mask(cand_leng)
            #logit:batch cand_len
            logit.masked_fill_(candidate_mask, -float('inf'))

            # Supervised training 
            #target:batch
            target = self._teacher_action(perm_obs, ended)
            #IL training
            ml_loss += self.criterion(logit, target)

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                # teacher forcing
            elif self.feedback == 'argmax': 
                _, a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
                #probability
                log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
                #log_probs:batch
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))   # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)    # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())      # For log
                entropys.append(c.entropy())                                # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (cand_leng[i]-1) or next_id == args.ignoreid or ended[i]:    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            # Make action and get the new state
            #cand == action
            #len(a_t) == batch
            input_a_t = []
            for i,a in enumerate(cpu_a_t):
                #not end
                if a!=-1:
                    input_a_t.append(cand_angle_feat[i,a,:])
                else:
                    input_a_t.append(torch.zeros(args.angle_feat_size).cuda())
            #input_a_t:batch angle_feat_size
            input_a_t = torch.stack(input_a_t)
            self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)
            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]                    # Perm the obs for the resu

            if train_rl:
                # Calculate the mask and reward
                dist = np.zeros(batch_size, np.float32)
                ndtw_score = np.zeros(batch_size, np.float32)
                reward = np.zeros(batch_size, np.float32)
                mask = np.ones(batch_size, np.float32)
                for i, ob in enumerate(perm_obs):
                #for each agent
                    dist[i] = ob['distance']
                    #get path 
                    path_act = [vp[0] for vp in traj[i]['path']]
                    #cal ndtw between path and gt_path
                    ndtw_score[i] = self.ndtw_criterion[ob['scan']](path_act, ob['gt_path'], metric='ndtw')

                    if ended[i]:            # If the action is already finished BEFORE THIS ACTION.
                        reward[i] = 0.
                        mask[i] = 0.
                    else:       # Calculate the reward
                        action_idx = cpu_a_t[i]
                        if action_idx == -1:        # If the action now is end
                            if dist[i] < 3:         # Correct
                                #reward[i] = 2.
                                reward[i] = 2.0 + ndtw_score[i] * 2.0
                            else:                   # Incorrect
                                reward[i] = -2.
                        else:                       # The action is not end
                            reward[i] = - (dist[i] - last_dist[i])      # Change of distance
                            ndtw_reward = ndtw_score[i] - last_ndtw[i]
                            if reward[i] > 0:                           # Quantification
                                #reward[i] = 1
                                reward[i] = 1 + ndtw_reward
                            elif reward[i] < 0:
                                #reward[i] = -1
                                reward[i] = -1 + ndtw_reward
                            else:
                                raise NameError("The action doesn't change the move")
                            if (last_dist[i] <= 1.0) and (dist[i]-last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                #reward:batch
                rewards.append(reward)
                #mask:batch
                masks.append(mask)
                #dist:batch
                last_dist[:] = dist
                #ndtw:batch
                last_ndtw[:] = ndtw_score

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            #ened || (cpu_a_t == -1)
            #ended:batch
            #cpu_at:batch -1 means ending
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all(): 
                #break episode iter
                break
        #episode over calculate loss
        if train_rl:
            # Last action in A2C
            _, f_t, \
            cand_leng, cand_visual_feat, cand_angle_feat, cand_obj_feat, \
            near_visual_mask, near_visual_feat, near_angle_feat, \
            near_obj_mask, near_obj_feat, near_edge_feat, near_id_feat  = self.get_input_feat(perm_obs)
            if speaker is not None:       # Apply the env drop mask to the feat
                f_t[..., :-args.angle_feat_size] *= noise
                cand_visual_feat *= noise
                near_visual_feat *= noise

            last_h_, _, _, _, _,scorer,_ = self.decoder(input_a_t, f_t,
                                            cand_visual_feat, cand_angle_feat, cand_obj_feat,
                                            near_visual_mask, near_visual_feat, near_angle_feat,
                                            near_obj_mask, near_obj_feat, near_edge_feat,near_id_feat,  
                                            h_t, h1, c_t,
                                            ctx, ctx_mask,
                                            already_dropfeat=(speaker is not None))
            rl_loss = 0.
            # NOW, A2C!!!
            # Calculate the final discounted reward
            #last_h_:batch rnn_dim
            last_value__ = self.critic(last_h_).detach()    # The value esti of the last state, remove the grad for safety
            #scorer:batch gcn_dim
            last_value_object = self.critic_object(scorer).detach()
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            discount_reward_object = np.zeros(batch_size, np.float32)
            for i in range(batch_size):
                if not ended[i]:        # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]
                    discount_reward_object[i] = last_value_object[i]
            length = len(rewards)
            total = 0
            #inverse iter
            #for each time step
            for t in range(length-1, -1, -1):
                #rewards[t]:batch 
                discount_reward = discount_reward * args.gamma + rewards[t]   # If it ended, the reward will be 0
                discount_reward_object = discount_reward_object *args.gamma + rewards[t]
                mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).cuda()
                clip_reward = discount_reward.copy()
                clip_reward_object = discount_reward_object.copy()
                r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).cuda()
                r_object = Variable(torch.from_numpy(clip_reward_object),requires_grad=False).cuda()
                #hidden_states[t]:batch rnn_dim
                #v : batch 
                v_ = self.critic(hidden_states[t])
                v_object = self.critic_object(scorers[t])
                #advantage
                a_ = (r_ - v_).detach()
                a_object = (r_object -v_object).detach()
                # r_: The higher, the better. -ln(p(action)) * (discount_reward - value)
                rl_loss +=(-policy_log_probs[t] * a_ * mask_).sum()
                rl_loss +=0.2*(-policy_score_probs[t]*a_object*mask_).sum()
                rl_loss +=(((r_ - v_) ** 2) * mask_).sum() * 0.5     # 1/2 L2 loss
                rl_loss +=0.2*(((r_object - v_object)**2)*mask_).sum()*0.5
                #entropy:batch p * logp
                if self.feedback == 'sample':
                    rl_loss += (-0.01 * entropys[t] * mask_).sum()
                    rl_loss += (-0.01 *entropys_object[t] *mask_).sum()
                self.logs['critic_loss'].append((((r_ - v_) ** 2) * mask_).sum().item())
                #number of not ended agent
                total = total + np.sum(masks[t])
            self.logs['total'].append(total)
            # Normalize the loss function
            if args.normalize_loss == 'total':
                rl_loss /= total
            elif args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert args.normalize_loss == 'none'

            self.loss += rl_loss
            self.logs['RL_loss'].append(rl_loss.item())

        if train_ml is not None:
            self.loss += ml_loss * train_ml / batch_size
            self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.episode_len)    # This argument is useless.

        return traj

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            for model in self.models:
                model.train()
        else:
            for model in self.models:
                model.eval()
        super(Seq2SeqAgent, self).test(iters)

    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

    def accumulate_gradient(self, feedback='teacher', **kwargs):
        if feedback == 'teacher':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
        elif feedback == 'sample':
            self.feedback = 'teacher'
            self.rollout(train_ml=params['mlWeight'], train_rl=False, **kwargs)
            self.feedback = 'sample'
            self.rollout(train_ml=None, train_rl=True, **kwargs)
        else:
            assert False

    def optim_step(self):
        self.loss.backward()

        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)

        for optimizer in self.optimizer:
            optimizer.step()

    def train(self, n_iters, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback

        for model in self.models:
            model.train()

        self.losses = []
        for iter in range(1, n_iters + 1):
            for optimizer in self.optimizers:
                optimizer.zero_grad()

            self.loss = 0
            if feedback == 'teacher':
                self.feedback = 'teacher'
                self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
            elif feedback == 'sample':
                if params['mlWeight'] != 0:
                    self.feedback = 'teacher'
                    self.rollout(train_ml=params['mlWeight'], train_rl=False, **kwargs)
                self.feedback = 'sample'
                self.rollout(train_ml=None, train_rl=True, **kwargs)
            else:
                assert False

            self.loss.backward()

            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
            torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)

            for optimizer in self.optimizers:
                optimizer.step()

            if args.aug is None:
                utils.print_progress(iter, n_iters+1, prefix='Progress:', suffix='Complete', bar_length=50)
                
    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [
            # ("encoder",self.encoder,self.encoder_optimizer),
            ("decoder",self.decoder,self.decoder_optimizer),
            ("critic",self.critic,self.critic_optimizer),
            # ("critic_object",self.critic_object,self.critic_object_optimizer),
            # ("objencoder",self.objencoder,self.objencoder_optimizer)
        ]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [
            # ("encoder",self.encoder,self.encoder_optimizer),
            ("decoder",self.decoder,self.decoder_optimizer),
            ("critic",self.critic,self.critic_optimizer),
            # ("critic_object",self.critic_object,self.critic_object_optimizer),
            # ("objencoder",self.objencoder,self.objencoder_optimizer)
        ]
        for param in all_tuple:
            recover_state(*param)
        return states['decoder']['epoch'] - 1

