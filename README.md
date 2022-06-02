## Release Note

CBAM and mixedstyle will be merged soon. Everything else has been uploaded.


## Install

Follow the official [repo](https://github.com/ifzhang/ByteTrack) to install bytetrack.


## Data Prepare

We used the MOTSynth official data extraction [pipelines](https://github.com/dvl-tum/motsynth-baselinesz).

```
datasets
   |——————mot (MOT17)
   |        └——————train
   |        └——————test
   └——————motsynth
   |         └——————MOT17-02-DPM
   |         └——————MOT17-04-DPM
   |         └——————...
   |         └——————annotations
   |         └——————comb_annotations
   |         └——————frames
   └——————data_path
```
![image](https://user-images.githubusercontent.com/33244972/171125695-38f0b3e7-2a47-42c4-9740-18f530919a2b.png)

## Training

To reproduce the performance, you need 8 GPUs with no less than 40G memory.

- **Stage1.** Training warm_up model with below script, or download [warm-up model](https://drive.google.com/drive/folders/1edc3XEYMQlVSWkuEiGYyBdUAKI5MYz2O?usp=sharing) (58.1 HOTA), and save it in
```
python3 tools/train.py -f exps/example/mot/yolox_x_source_only.py -d 8 -b 48 --fp16 -o
```
- Make pseudo label, run below code 
```
python3 tools/track.py -f exps/example/mot/yolox_x_mix_det.py -c weight/warm-up.pth.tar -b 1 -d 1 --fp16 --fuse
python3 tools/interpolation.py
python3 make_PU.py
python3 ./tools/convert_mot17_to_coco_pu.py
python3 merge_json.py
```

- **Stage2.** Cross-domain Mixed Sampling with mosaic augmentation
```
python3 tools/train.py -f exps/example/mot/yolox_x_mixed.py -d 8 -b 48 --fp16 -o -c weight/warm-up.pth.tar
```
- Make pseudo label, run below code 
```
python3 tools/track.py -f exps/example/mot/yolox_x_ft.py -c weight/mixed.pth.tar -b 1 -d 1 --fp16 --fuse
python3 tools/interpolation.py
python3 make_PU.py
python3 ./tools/convert_mot17_to_coco_pu.py # We removed values with confidence less than 0.7 (L 108 in ./tools/convert_mot17_to_coco_pu.py) because predictions with low confidence can act as label noise.
``` 
- **Stage3.** Make multiple fine-tune model and model soup # when fine-tuned, the EMA is not used.

(Note that when performing fine-tune in Step 3, the augmentation combination should be different in L49-57 of ./yolox/data/datasets/mot.py)

```
python3 wa.py # you have to adjust it manually. (Until the CVPR22 conference, the completed code will be uploaded.)
```

## Test

```
python3 tools/track.py -f exps/example/mot/yolox_x_source_only.py -c weight/warm-up_67.5.pth.tar -b 1 -d 1 --fp16 --fuse
python3 tools/interpolation.py # HOTA 58.1
```
![image](https://user-images.githubusercontent.com/33244972/171542787-c3a84452-b54c-4fb6-919e-cccef222153d.png)

```
python3 tools/track.py -f exps/example/mot/yolox_x_source_only.py -c weight/stage2_69.6.pth.tar -b 1 -d 1 --fp16 --fuse
python3 tools/interpolation.py # HOTA 59.xx
```
![image](https://user-images.githubusercontent.com/33244972/171543869-2cb71762-f612-485c-aed3-ee6d54c2fc5c.png)

```
python3 tools/track.py -f exps/example/mot/yolox_x_source_only.py -c weight/stage3_75.7.pth.tar -b 1 -d 1 --fp16 --fuse
python3 tools/interpolation.py # HOTA 62.xx
```
![image](https://user-images.githubusercontent.com/33244972/171544560-3063f81e-075a-4a06-a09d-f899fff29d46.png)

```
python3 tools/track.py -f exps/example/mot/yolox_x_source_only.py -c weight/stage3_77.9.pth.tar -b 1 -d 1 --fp16 --fuse
python3 tools/interpolation.py # HOTA 63.xx
```
![image](https://user-images.githubusercontent.com/33244972/171545301-f356b3e2-03dd-4561-a22b-b2eae722e895.png)
