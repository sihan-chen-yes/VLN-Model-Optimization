name=train_clip_obj_r2r_origin
flag="--attn soft --train listener
      --featdropout 0.3
      --visual_feat --angle_feat
      --feedback sample
      --mlWeight 0.2 
      --features clip --gcn_topk 5 --static_gcn_weights --CLIP_language --obj_clip
      --glove_dim 300 --top_N_obj 8 --distance_decay_function same  
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 80000 --maxAction 20"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 python3 methods/SEvol/train.py $flag --name $name 
