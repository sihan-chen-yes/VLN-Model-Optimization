''' Batched Room-to-Room navigation environment '''

import sys
sys.path.append('buildpy36')
#import MatterSim
import csv
import numpy as np
import math
import base64
import utils
import json
import os
import random
import networkx as nx
from param import args
from collections import namedtuple, defaultdict
from utils import load_datasets, load_nav_graphs, Tokenizer
angle_inc = np.pi / 6.
ANGLE_INC = np.pi / 6.
def structured_map(function, *args, **kwargs):
    #assert all(len(a) == len(args[0]) for a in args[1:])
    nested = kwargs.get('nested', False)
    acc = []
    for t in zip(*args):
        if nested:
            mapped = [function(*inner_t) for inner_t in zip(*t)]
        else:
            mapped = function(*t)
        acc.append(mapped)
    return acc
csv.field_size_limit(sys.maxsize)
WorldState = namedtuple("WorldState", ["scanId", "viewpointId","viewIndex","heading", "elevation","x","y","z"])
class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments, 
        using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store=None, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feature_store: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        if feature_store:
            if type(feature_store) is dict:     # A silly way to avoid multiple reading
                self.features = feature_store
                self.image_w = 640
                self.image_h = 480
                self.vfov = 60
                self.feature_size = next(iter(self.features.values())).shape[-1]
                print('The feature size is %d' % self.feature_size)
        else:
            print('Image features not provided')
            self.image_w = 640
            self.image_h = 480
            self.vfov = 60
            self.features = feature_store
            self.feature_size = 2048
        if args.debug:
            self.featurized_scans = set(json.load(open('./methods/neural_symbolic/debug_featurized_scans.json','r')))
            self.feature_size = 512
        else:
            self.featurized_scans = set([key.split("_")[0] for key in list(self.features.keys())])
        adj_dict = json.load(open("./datasets/total_adj_list.json",'r'))
        self.adj_dict = adj_dict
        self.graphs = load_nav_graphs(self.featurized_scans)
        assert adj_dict is not None, "Error! No adjacency dictionary!"


    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId   

    def newEpisodes(self, scanIds, viewpointIds, headings, elevation=None):
        def f(scan_id, viewpoint_id, heading):
            position =  self.graphs[scan_id].node[viewpoint_id]['position']
            elevation = 0
            view_index = (12 * round(elevation / ANGLE_INC + 1)
                          + round(heading / ANGLE_INC) % 12)
            
            return WorldState(scanId=scan_id,
                              viewpointId=viewpoint_id,
                              viewIndex=view_index,
                              heading=heading,
                              elevation=elevation,
                              x = position[0],
                              y = position[1],
                              z = position[2])
        self.sims = structured_map(f, scanIds, viewpointIds, headings)
  
    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((30, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sim in enumerate(self.sims):
            long_id = self._make_id(sim.scanId, sim.viewpointId)
            if self.features is not None:
                feature = self.features[long_id]
                # feature = self.features[long_id][state.viewIndex,:]
                feature_states.append((feature, sim))
            else:
                feature_states.append((None, sim))
        return feature_states

    def get_adjs(self, world_states):
        def f(world_state):
            query = '_'.join([world_state.scanId,
                              world_state.viewpointId,
                              str(world_state.viewIndex)])
            return self.adj_dict[query]
        return structured_map(f, world_states)

    def makeActions(self, actions, attrs=None):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        if attrs is None:
            attrs = self.get_adjs(self.sims)
        def f(world_state, action, loc_attrs):
            if action == 0 or action < 0:
                return world_state
            else:
                loc_attr = loc_attrs[action]
                location = self.graphs[world_state.scanId].node[loc_attr['nextViewpointId']]['position']
                return WorldState(scanId=world_state.scanId,
                                  viewpointId=loc_attr['nextViewpointId'],
                                  viewIndex=loc_attr['absViewIndex'],
                                  heading=(
                                      loc_attr['absViewIndex'] % 12) * ANGLE_INC,
                                  elevation=(
                                      loc_attr['absViewIndex'] // 12 - 1)
                                  * ANGLE_INC,
                                  x = location[0],
                                  y = location[1],
                                  z = location[2])
        world_states = self.sims
        world_states =  structured_map(f, world_states, actions, attrs)
        self.sims = world_states

class Simulator(object):

    def __init__(self, scans = None):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feat_db: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        #TODO copy the adj_file from 98
        adj_dict = json.load(open("./datasets/total_adj_list.json",'r'))
        self.adj_dict = adj_dict
        self.graphs = load_nav_graphs(scans)
        #self.label_set = json.load(open('data/labels/all_labels.json'))
        #self.label_set = load_datas()
        #self.room_label = json.load(open('data/labels/reverie_room/house_pano_info.json'))
        assert adj_dict is not None, "Error! No adjacency dictionary!"

    # def _make_id(self, scanId, viewpointId):
    #     return scanId + '_' + viewpointId

    def newEpisode(self, scan_id, viewpoint_id, heading):
        elevation = 0
        view_index = (12 * round(elevation / ANGLE_INC + 1)
                        + round(heading / ANGLE_INC) % 12)
        location = self.graphs[scan_id].node[viewpoint_id]['position']
        self.state =  WorldState(scanId=scan_id,
                            viewpointId=viewpoint_id,
                            viewIndex=view_index,
                            heading=heading,
                            elevation=elevation,
                            x = location[0],
                            y = location[1],
                            z = location[2])
    # def getState(self):
    #     """
    #     Get list of states augmented with precomputed image features. rgb field will be empty.
    #     Agent's current view [0-35] (set only when viewing angles are discretized)
    #         [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
    #     :return: [ ((36, 2048), sim_state) ] * batch_size
    #     """
    #     return self.state
    def get_adjs(self, world_state=None):
        if world_state ==None:
            world_state = self.state
        query = '_'.join([world_state.scanId,
                            world_state.viewpointId,
                            str(world_state.viewIndex)])
        return self.adj_dict[query]
    # def makeActions(self, action, loc_attrs=None):
    #     ''' Take an action using the full state dependent action interface (with batched input).
    #         Every action element should be an (index, heading, elevation) tuple. '''
    #     if loc_attrs is None:
    #        loc_attrs =  self.get_adjs(self.state)
    #     if action == 0:
    #         return world_state
    #     else:
    #         loc_attr = loc_attrs[action]
    #         location = self.graphs[world_state.scanId][loc_attr['nextViewpointId']]['position']
    #         self.state = WorldState(scanId=world_state.scanId,
    #                             viewpointId=loc_attr['nextViewpointId'],
    #                             viewIndex=loc_attr['absViewIndex'],
    #                             heading=(
    #                                 loc_attr['absViewIndex'] % 12) * ANGLE_INC,
    #                             elevation=(
    #                                 loc_attr['absViewIndex'] // 12 - 1)
    #                             * ANGLE_INC,
    #                             x = location[0],
    #                             y = location[1],
    #                             z = location[2])


class R2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store, obj_store, batch_size=100, seed=10, splits=['train'], tokenizer=None,
                 name=None):
        self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size)
        self.obj_dict = obj_store
        if feature_store:
            self.feature_size = self.env.feature_size
        if args.debug:
            self.feature_size=2048
        self.data = []
        if tokenizer:
            self.tok = tokenizer
        scans = []
        for split in splits:
            for item in load_datasets([split]):
                # Split multiple instructions into separate entries
                for j,instr in enumerate(item['instructions']):
                    if item['scan'] not in self.env.featurized_scans:   # For fast training
                        continue
                    new_item = dict(item)
                    if args.task =='R2R':
                        id_ = item['path_id']
                    else:
                        id_ = item['id']
                    new_item['instr_id'] = '%s_%d' % (id_, j)
                    new_item['instructions'] = instr
                    if tokenizer:
                        new_item['instr_encoding'] = tokenizer.encode_sentence(instr)
                    if not tokenizer or new_item['instr_encoding'] is not None:  # Filter the wrong data
                        self.data.append(new_item)
                        scans.append(item['scan'])
        if name is None:
            self.name = splits[0] if len(splits) > 0 else "FAKE"
        else:
            self.name = name

        self.scans = set(scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()

        self.angle_feature = utils.get_all_point_angle_feature()
        self.sim = Simulator(self.scans)
        self.buffered_state_dict = {}

        # near point
        self.pid2near_pid = np.zeros([36,5], dtype=np.int32)
        self.pid2angle = np.zeros([36,2], dtype=np.float32)
        for c in range(36):
            l = c+11 if c%12==0 else c-1
            r = c-11 if c%12==11 else c+1
            t = -1 if c//12==2 else c+12
            b = -1 if c//12==0 else c-12
            self.pid2near_pid[c,:] = np.array([c,l,t,r,b],dtype=np.int32)
            self.pid2angle[c,0] = (c%12)*math.radians(30)
            self.pid2angle[c,1] = (c//12)*math.radians(30)+math.radians(-30)
        self.pid2near_mask = (self.pid2near_pid == -1)

        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, tile_one=False, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        if tile_one:
            batch = [self.data[self.ix]] * batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix: self.ix+batch_size]
            if len(batch) < batch_size:
                random.shuffle(self.data)
                self.ix = batch_size - len(batch)
                batch += self.data[:self.ix]
            else:
                self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing. 
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.viewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.paths[state.scanId][state.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc['rel_heading'] ** 2 + loc['rel_elevation'] ** 2)
        def _get_near(v, base_heading):
            def _np_angle_feature(heading, elevation):
                e_heading = np.expand_dims(heading, axis=1)
                e_elevation = np.expand_dims(elevation, axis=1)
                N = args.angle_feat_size // 4 # repeat time
                return np.concatenate([np.sin(e_heading).repeat(N,1), np.cos(e_heading).repeat(N,1), np.sin(e_elevation).repeat(N,1), np.cos(e_elevation).repeat(N,1)], -1)
            c = v['pointId']
            cand_heading = v['heading']
            cand_elevation = v['elevation']
            near_pointId = self.pid2near_pid[c,:]
            near_mask = self.pid2near_mask[c,:]
            default_near_pointId = near_pointId.copy()
            default_near_pointId[near_mask==True] = 0

            near_heading = self.pid2angle[default_near_pointId,0]-base_heading
            near_elevation = self.pid2angle[default_near_pointId,1]
            near_rel_heading = near_heading-cand_heading
            near_rel_elevation = near_elevation-cand_elevation

            near_visual_feat = feature[default_near_pointId]
            near_angle_feat = _np_angle_feature(near_heading, near_elevation)
            near_edge_feat = _np_angle_feature(near_rel_heading, near_rel_elevation)
            near_obj_class = [self.obj_dict[scanId][viewpointId][pointId]['object_class'][:args.top_N_obj] 
                                if pointId != -1 else np.zeros(args.top_N_obj) for pointId in near_pointId]
            near_obj_class = np.stack(near_obj_class)
            near_view_id = np.array(near_pointId)
            return near_pointId, near_mask, near_visual_feat, near_angle_feat, near_edge_feat, near_obj_class,near_view_id
            
        base_heading = (viewId % 12) * math.radians(30)
        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        # if long_id not in self.buffered_state_dict:
        #     for ix in range(36):
        #         if ix == 0:
        #             self.sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
        #         elif ix % 12 == 0:
        #             self.sim.makeAction([0], [1.0], [1.0])
        #         else:
        #             self.sim.makeAction([0], [1.0], [0])

        #         state = self.sim.getState()[0]
        #         assert state.viewIndex == ix

        #         # Heading and elevation for the viewpoint center
        #         heading = state.heading - base_heading
        #         elevation = state.elevation

        #         visual_feat = feature[ix]

                # get adjacent locations
        self.sim.newEpisode(scanId, viewpointId, base_heading)
        navigableLocations = self.sim.get_adjs()
        for j, loc in enumerate(navigableLocations[1:]):
            # if a loc is visible from multiple view, use the closest
            # view (in angular distance) as its representation
            #distance = _loc_distance(loc)

            # Heading and elevation for for the loc
            loc_heading = loc['rel_heading']
            loc_elevation = loc['rel_elevation']
            angle_feat = utils.angle_feature(loc_heading, loc_elevation)
            visual_feat = feature[loc['absViewIndex']]
            adj_dict[loc['nextViewpointId']] = {
                'heading': loc_heading,
                'elevation': loc_elevation,
                "normalized_heading": base_heading + loc_heading,
                'scanId':scanId,
                'viewpointId': loc['nextViewpointId'], # Next viewpoint id
                'pointId': loc['absViewIndex'],
                #'distance': distance,
                'idx': j + 1,
                'angle_feat': angle_feat,
                'visual_feat': visual_feat,
            }
        for k,v in adj_dict.items():
            adj_dict[k]['obj_class'] = self.obj_dict[scanId][viewpointId][v['pointId']]['object_class'][:args.top_N_obj]
            near_pointId, near_mask, near_visual_feat, near_angle_feat, near_edge_feat, near_obj_class,near_view_id = _get_near(v, base_heading)
            adj_dict[k]['near_pointId'] = near_pointId
            adj_dict[k]['near_mask'] = near_mask
            adj_dict[k]['near_visual_feat'] = near_visual_feat
            adj_dict[k]['near_angle_feat'] = near_angle_feat
            adj_dict[k]['near_edge_feat'] = near_edge_feat    
            adj_dict[k]['near_obj_class'] = near_obj_class
            adj_dict[k]['near_view_id'] = near_view_id

        candidate = list(adj_dict.values())
        self.buffered_state_dict[long_id] = [
            {key: c[key]
                for key in
                ['normalized_heading', 'elevation', 'scanId', 'viewpointId',
                    'pointId', 'idx']}
            for c in candidate
        ]
        return candidate
   
    def _get_obs(self):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            base_view_id = state.viewIndex

            # Full features
            candidate = self.make_candidate(feature, state.scanId, state.viewpointId, state.viewIndex)

            # (visual_feature, angel_feature) for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)
            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'candidate': candidate,
                'navigableLocations' : self.sim.get_adjs(state),
                'instructions' : item['instructions'],
                'teacher' : self._shortest_path_action(state, item['path'][-1]),
                'gt_path': item['path'],
                'path_id' : item['path_id']
            })
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
            # A2C reward. The negative distance between the state and the final state
            obs[-1]['distance'] = self.distances[state.scanId][state.viewpointId][item['path'][-1]]
        return obs

    def reset(self, batch=None, inject=False, **kwargs):
        ''' Load a new minibatch / episodes. '''
        if batch is None:       # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        else:
            if inject:          # Inject the batch into the next minibatch
                self._next_minibatch(**kwargs)
                self.batch[:len(batch)] = batch
            else:               # Else set the batch to the current batch
                self.batch = batch
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    def get_statistics(self):
        stats = {}
        length = 0
        path = 0
        for datum in self.data:
            length += len(self.tok.split_sentence(datum['instructions']))
            path += self.distances[datum['scan']][datum['path'][0]][datum['path'][-1]]
        stats['length'] = length / len(self.data)
        stats['path'] = path / len(self.data)
        return stats


