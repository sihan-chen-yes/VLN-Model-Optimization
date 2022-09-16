
import torch
def length2mask(length, size=None):
    batch_size = len(length)
    size = int(max(length)) if size is None else size
    mask = (torch.arange(size, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
                > (torch.LongTensor(length) - 1).unsqueeze(1)).cuda()
    return mask

if __name__ == "__main__":
    mask = torch.ones(5).cuda()
    scores = torch.arange(5).cuda()
    if mask is not None:
        print(mask.bool())
        scores.masked_fill_(mask.bool(), -1)
    print(scores)