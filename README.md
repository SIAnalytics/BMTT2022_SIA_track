## Training

To reproduce the performance, you need 8 GPUs with no less than 40G memory.

- **Stage1.** Training warm_up model with below script, or download [warm-up model](-) (58.1 HOTA), and save it in
```
python3 tools/train.py -f exps/example/mot/yolox_x_warm_up.py -d 8 -b 48 --fp16 -o
```
- Make pseudo label, run below code 
```
python3 tools/track.py -f exps/example/mot/yolox_x_mix_det.py -c pretrained/warm-up.pth.tar -b 1 -d 1 --fp16 --fuse
python3 tools/interpolation.py
python3 make_PU.py
python3 ./tools/convert_mot17_to_coco_pu.py
```

- **Stage2.** Cross-domain Mixed Sampling with mosaic augmentation
```
python3 tools/train.py -f exps/example/mot/yolox_x_mixed_training.py -d 8 -b 48 --fp16 -o -c pretrained/warm-up.pth.tar
```
- Make pseudo label, run below code 
```
python3 tools/track.py -f exps/example/mot/yolox_x_mix_det.py -c pretrained/mixed.pth.tar -b 1 -d 1 --fp16 --fuse
python3 tools/interpolation.py
python3 make_PU.py
python3 ./tools/convert_mot17_to_coco_pu.py
```
- **Stage3.** Make multiple fine-tune model and model soup
```
TODO

```
