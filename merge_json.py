import json

#{'file_name': 'frames/042/rgb/0001.jpg', 'id': 420000, 'frame_n': 1, 'cam_world_pos': [-355.121, -65.8944, -781.963], 'cam_world_rot': [-45.5578, 25.9048, 0.0542865], 'ignore_mask': [0, 0, 0, 0], 'height': 1080, 'width': 1920, 'description': 'MOTSynth 2021 Dataset - Sequence #42', 'version': '1.0', 'img_height': 1080, 'img_width': 1920, 'seq_name': '042', 'is_night': 0, 'weather': 'CLEAR', 'is_moving': 0, 'cam_fov': 50.0, 'fps': 20, 'sequence_length': 1800, 'time': '11', 'fx': '1158', 'fy': '1158', 'cx': '960', 'cy': '540'}

# {'file_name': 'MOT17-13-FRCNN/img1/000742.jpg', 'id': 5308, 'frame_id': 742, 'prev_image_id': 5307, 'next_image_id': 5309, 'video_id': 7, 'height': 1080, 'width': 1920}

labeled_source_datat = './datasets/motsynth/comb_annotations/train.json'
pseudo_labeled_target_data = './datasets/mot/annotations/train.json'


with open(pseudo_labeled_target_data, 'r') as f:
    pu_t = json.load(f)

with open(labeled_source_datat, 'r') as f:
    l_s = json.load(f)

for i in pu_t['images']:
    i['id'] += 7591790
    i['frame_n']=[]
    i['cam_world_pos'] = []
    i['cam_world_rot'] = []
    i['ignore_mask'] = []
    i['height'] = 1080
    i['width'] = 1920
    i['description'] = []
    i['version'] = []
    i['img_height'] = 1080
    i['img_width'] = 1920
    i['seq_name'] = 'mot'
    i['is_night'] = 0
    i['weather'] = 'CLEAR'
    i['is_moving'] = 0
    i['cam_fov'] = 50.0
    i['fps'] = 20
    i['sequence_length'] = 2000
    i['time'] = '11'
    i['fx'] = '1158'
    i['fy'] = '1158'
    i['cx'] = '960'
    i['cy'] = '540'
    i.pop('prev_image_id')
    i.pop('next_image_id')
    i.pop('video_id')
    l_s['images'].append(i)


for i in pu_t['annotations']:
    i['id'] += 7591790046917
    i['image_id'] += 7591790
    i['segmentation'] = {}
    i['keypoints'] = []
    i['keypoints_3d'] = []
    i['num_keypoints'] = []
    i['ped_id'] = []
    i['model_id'] = []
    i['attributes'] = []
    i['is_blurred'] = []
    l_s['annotations'].append(i)

with open("mixed_sample.json", "w") as json_file:
    json.dump(l_s, json_file)

