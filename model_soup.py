import torch
import torch.nn as nn

ckpt = torch.load('72.6.pth.tar', map_location='cpu')
ckpt2 = torch.load('73.6.pth.tar', map_location='cpu')

for k, v in ckpt["model"].items():
    f = ckpt2["model"][k]*0.5 + v*0.5
    ckpt2["model"][k] = f

torch.save(ckpt2, 'soup.pth.tar')
