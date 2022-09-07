from numpy import angle
import torch
import clip
import pickle as pkl
from env import *
from utils import *
def gen_obj_embedding(clip_model_name,file_name):
    """gen obj_embedding matrix"""
    """clip_model_name: RN101, RN50, RN50x4, ViT-B/32"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(clip_model_name, device=device)
    with open('img_features/objects/object_vocab.txt', 'r') as f_ov:
        obj_vocab = [k.strip() for k in f_ov.readlines()]
        obj_vocab.append("forward")
        obj_vocab.append(".")
    text = clip.tokenize(obj_vocab).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
    obj_clip = {}
    _,feat_size = text_features.size()
    for i in range(len(obj_vocab)):
        obj_clip[obj_vocab[i]] = text_features[i].cpu().numpy()
    print(feat_size)
    with open('img_features/objects/' + file_name +'.pkl', 'wb') as f:
        pickle.dump(obj_clip,f)
    print("finish generating obj_embedding matrix")

def gen_angle_feat():
    """record all_point_angle_feature of simulator"""
    TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
    CLIP_FEATURES = 'img_features/CLIP-ResNet-50x4-views.tsv'
    features = CLIP_FEATURES
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=77)
    feat_dict = read_img_features(features)
    with open('./img_features/objects/pano_object_class.pkl', 'rb') as f:
        obj_dict = pkl.load(f)
    
    train_env = R2RBatch(feat_dict,obj_dict,batch_size=64,
                splits=['val_unseen'], tokenizer=tok)
    with open('img_features/all_point_angle_feature_tmp.pkl', 'wb') as f:
        pickle.dump(train_env.angle_feature,f)
    print("finish recording all_point_angle_feature")

def read_angle_feat():
    with open('img_features/all_point_angle_feature.pkl', 'rb') as f:
        angle_feature = pickle.load(f)
    with open('img_features/all_point_angle_feature_tmp.pkl', 'rb') as f:
        angle_feature_tmp = pickle.load(f)
    for i in range(len(angle_feature)):
        assert len(angle_feature) == len(angle_feature_tmp)
        if angle_feature[i].all() != angle_feature_tmp[i].all():
            print("not same")
            break
    print("same")

if __name__ == '__main__':
    read_angle_feat()