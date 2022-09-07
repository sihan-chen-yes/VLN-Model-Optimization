name=train_clip_obj_r2r_SEM_50x4_noenc_valid_simplified
flag="--attn soft --train validlistener
      --featdropout 0.3 
      --visual_feat --angle_feat
      --feedback sample
      --mlWeight 0.2 
      --features clip --gcn_dim 128 --visual_feat_size 640 --clip_dim 640 --rnnDim 640 --gcn_topk 5 --static_gcn_weights --CLIP_language --obj_clip
      --top_N_obj 8 --distance_decay_function same  
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 80000 --maxAction 20
      --load snap/train_clip_obj_r2r_SEM_50x4_noenc/state_dict/best_val_unseen"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 python3 methods/SEvol/train.py $flag --name $name 

# Try this with file logging:
# CUDA_VISIBLE_DEVICES=$1 unbuffer python r2r_src/train.py $flag --name $name | tee snap/$name/logz
