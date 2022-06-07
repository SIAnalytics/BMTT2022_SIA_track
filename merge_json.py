import json
import copy

labeled_source_datat = './datasets/motsynth/comb_annotations/train.json'
pseudo_labeled_target_data = './datasets/mot/annotations/pseudo_train.json' # <- pseudo label path


with open(pseudo_labeled_target_data, 'r') as f:
    pu_t = json.load(f)

with open(labeled_source_datat, 'r') as f:
    l_s = json.load(f)

print('before Number of Images:', len(l_s['images']))
print('before Number of Instance:', len(l_s['annotations']))


for over in range(20):
    tmp_pu_t = copy.deepcopy(pu_t)
    for i in pu_t['images']:
        i['id'] += 7591790000 + over*len(pu_t['images'])# source dataset 
        i['frame_n']=[]
        i['cam_world_pos'] = []
        i['cam_world_rot'] = []
        i['ignore_mask'] = []
        #i['height'] = 1080
        #i['width'] = 1920
        i['description'] = []
        i['version'] = []
        i['img_height'] = i['height']
        i['img_width'] = i['width']
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

        l_s['images'].append(i)

    for i in pu_t['annotations']:
        i['id'] += 7591790046917 + over*len(pu_t['annotations'])
        i['image_id'] += 7591790000 + over*len(pu_t['images'])
        i['segmentation'] = {}
        i['keypoints'] = []
        i['keypoints_3d'] = []
        i['num_keypoints'] = 22
        i['ped_id'] = 1
        i['model_id'] = '1'
        i['attributes'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        i['is_blurred'] = 1
        l_s['annotations'].append(i)

with open("mixed_sample.json", "w") as json_file:
    json.dump(l_s, json_file)

print('after Number of Images:', len(l_s['images']))
print('after Number of Instance:',len(l_s['annotations']))

