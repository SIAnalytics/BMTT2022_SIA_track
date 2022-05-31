import torch
import torch.nn as nn
import argparse


parser = argparse.ArgumentParser(description='weight average')
parser.add_argument('--path1', type=str)
parser.add_argument('--path2', type=str)

args = parser.parse_args()

ckpt = torch.load(args.path1, map_location='cpu') # fine-tune model1
ckpt2 = torch.load(args.path2, map_location='cpu') # fine-tune model2

for k, v in ckpt["model"].items():
    f = ckpt2["model"][k]*0.4 + v*0.6
    ckpt2["model"][k] = f

torch.save(ckpt2, 'soup.pth.tar')
