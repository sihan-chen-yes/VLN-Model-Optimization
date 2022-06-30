import torch
import clip
import pickle
device = "cuda" if torch.cuda.is_available() else "cpu"
# model, _ = clip.load("RN101", device=device)
# model, _ = clip.load("RN50", device=device)
# model, _ = clip.load("RN50x4", device=device)
model, _ = clip.load("ViT-B/32", device=device)
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
with open('img_features/objects/obj_clip_ViT-B-32_dict.pkl', 'wb') as f:
    pickle.dump(obj_clip,f)



