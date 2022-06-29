search_space = {
    'lr': {'_type': 'choice', '_value': [0.00005, 0.00014, 0.00023, 0.00032, 0.00041, 0.0005]},
    'batchSize': {'_type': 'choice', '_value': [32, 48, 64, 96]},
    'gcn_dim': {'_type': 'choice', '_value': [64, 128, 256]},
    'mlWeight': {'_type': 'choice', '_value': [0.05, 0.2, 0.5]}
}
from nni.experiment import Experiment

experiment = Experiment('local')

experiment.config.experiment_name = 'changed_SEM_NNI'

experiment.config.trial_command = 'python3 methods/SEvol/train.py ' \
                                  '--attn soft --train listener ' \
                                  '--featdropout 0.3 ' \
                                  '--visual_feat --angle_feat --feedback sample --mlWeight 0.2 --features clip ' \
                                  '--gcn_dim 128 --gcn_topk 5 --static_gcn_weights --CLIP_language --obj_clip ' \
                                  '--top_N_obj 8 --distance_decay_function same ' \
                                  '--subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 15000 --maxAction 20 ' \
                                  '--name changed_SEM_NNI'
experiment.config.trial_code_directory = '.'

experiment.config.search_space = search_space

experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

experiment.config.max_trial_number = 100
experiment.config.trial_concurrency = 4

experiment.config.training_service.use_active_gpu = True
experiment.config.training_service.max_trial_number_per_gpu = 1
# experiment.config.training_service.gpu_indices = [0, 1, 2, 3]

experiment.config.trial_gpu_number = None
# max_experiment_duration = '1h'
experiment.run(8080)
input('Press enter to quit')
experiment.stop()
# nni.experiment.Experiment.view()
