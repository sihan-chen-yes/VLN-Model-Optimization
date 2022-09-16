name=debug
flag="--attn soft --train listener
      --featdropout 0.3 --batchSize 8
      --visual_feat --angle_feat
      --feedback sample
      --mlWeight 0.2 
      --features clip --gcn_dim 128 --visual_feat_size 640
      --clip_dim 640 --rnnDim 640 
      --gcn_topk 5  --CLIP_language --obj_clip
      --top_N_obj 8 --distance_decay_function same 
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 80000 --maxAction 20"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 python3 methods/SEvol/train.py $flag --name $name 

# Try this with file logging:
# CUDA_VISIBLE_DEVICES=$1 unbuffer python r2r_src/train.py $flag --name $name | tee snap/$name/log
