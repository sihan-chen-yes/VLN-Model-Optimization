import torch
y = [[[1,2,3],[4,5,6]],[[7,8,9],[1,1,1]]]
#2,2,3
x = torch.tensor(y)
print(x.max(1))