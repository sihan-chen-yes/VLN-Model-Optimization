from numpy.core.fromnumeric import clip
import torch
import torch.nn as nn
text = ["Walk past the shower and the sink. Take the door on the left out into the bedroom. "
        "Exit the bedroom on the right. Walk past the large red and beige painting.Enter the bedroom with the balcony and stop beside the white chair with the blue pillow that's by the door. "]

from myclip import clip

with torch.no_grad():
    text = clip.tokenize(text, truncate=True).cuda()
    model, preprocess = clip.load('RN50x4')
    word_level_features, sent_level_features, max_length = model.encode_text(text)
print(word_level_features.size())
# seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
# seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]  # Full length
#
# seq_tensor = torch.from_numpy(seq_tensor)
# seq_lengths = torch.from_numpy(seq_lengths)
#
# # Sort sequences by lengths
# seq_lengths, perm_idx = seq_lengths.sort(0, True)  # True -> descending
# sorted_tensor = seq_tensor[perm_idx]
# mask = (sorted_tensor == padding_idx)[:, :seq_lengths[0]]