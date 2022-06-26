search_space = {
    'lr': {'_type': 'choice', '_value': [0.00005, 0.00014, 0.00023, 0.00032, 0.00041, 0.0005]},
    'batchSize': {'_type': 'choice', '_value': [16, 32, 48]},
}
from nni.experiment import Experiment
experiment = Experiment('local')
experiment.config.experiment_name = 'NNI_TEST'
#cmd for REVERIE
experiment.config.trial_command = 'CUDA_VISIBLE_DEVICES=1 python3 methods/SEvol-NNI/train.py ' \
                                  '--attn soft --train listener --task REVERIE ' \
                                  '--featdropout 0.3 ' \
                                  '--load snap/in_static_out_dy_rl+clip-valid-REVERIE+load/state_dict/best_val_unseen ' \
                                  '--visual_feat --angle_feat --feedback sample --mlWeight 0.2 --features clip ' \
                                  '--gcn_topk 5 --static_gcn_weights --CLIP_language --glove_dim 300 ' \
                                  '--top_N_obj 8 --distance_decay_function same ' \
                                  '--subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 2000 --maxAction 20 ' \
                                  '--name NNI-reverie'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.max_trial_number = 100
experiment.config.trial_concurrency = 1
# max_experiment_duration = '1h'
experiment.run(8080)
input('Press enter to quit')
experiment.stop()
# nni.experiment.Experiment.view()
