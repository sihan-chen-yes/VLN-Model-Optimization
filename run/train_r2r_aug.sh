name=train-r2r-aug
flag="--attn soft --train auglistener --selfTrain
      --aug tasks/R2R/data/aug_paths.json
      --speaker snap/speaker/state_dict/best_val_unseen_bleu
      --load snap/w_o_data_aug_best/state_dcit/best_val_unseen
      --visual_feat --angle_feat
      --gcn_topk 5 --glove_dim 300 --top_N_obj 8 --distance_decay_function same
      --accumulateGrad
      --featdropout 0.4
      --features clip
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 200000 --maxAction 20"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 python3 methods/SEvol/train.py $flag --name $name 

# Try this with file logging:
# CUDA_VISIBLE_DEVICES=$1 unbuffer python r2r_src/train.py $flag --name $name | tee snap/$name/log
