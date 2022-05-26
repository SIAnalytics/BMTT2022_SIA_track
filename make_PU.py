import glob
import shutil
import os

path = glob.glob("/workspace/PU/ByteTrack/YOLOX_outputs/yolox_x_mot17_half/track_results/*")
path2 = glob.glob('/workspace/PU/ByteTrack/datasets/mot/train/*/gt')

for i in path:
    for j in path2:
        if i.split('/')[-1][:-4] ==j.split('/')[-2]:
            print(os.path.join(j,'gt.txt'))
            shutil.copy(i, os.path.join(j,'gt.txt'))

