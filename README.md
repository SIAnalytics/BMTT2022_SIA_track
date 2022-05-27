## Training

To reproduce the performance, you need 8 GPUs with no less than 40G memory.

- **Stage1.** Download [warm-up model](-) (58.1 HOTA), and save it in
- Make pseudo label 

```
python3 tools/track.py -f exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar -b 1 -d 1 --fp16 --fuse
python3 tools/interpolation.py
python3 make_PU.py
python3 ./tools/convert_mot17_to_coco_pu.py
```

