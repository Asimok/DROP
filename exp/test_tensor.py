import torch

x1 = torch.randn(2, 3)
x2 = torch.randn(2, 3)

x3 = torch.cat((x1,x2),0)