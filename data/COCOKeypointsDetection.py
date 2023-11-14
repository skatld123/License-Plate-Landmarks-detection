import json
import math
import os
import os.path
import sys

import cv2
import numpy as np
import torch
import torch.utils.data as data
import albumentations as A

class COCOKeypointsDetection(data.Dataset):
    def __init__(self, json_path, img_dir, preproc=None, transform=None, type="train"):
        self.preproc = preproc
        self.imgs_path = []
        self.annotations = []
        self.transform = transform
        self.type = type
        with open(json_path, 'r') as json_file:
            coco_data = json.load(json_file)

            # Extract image paths and annotations from COCO format
            for image_info in coco_data['images']:
                self.imgs_path.append(os.path.join(img_dir, image_info['file_name']))
                image_id = image_info['id']
                image_annotations = []
                for anno in coco_data['annotations']:
                    if anno['image_id'] == image_id:
                        image_annotations.append(anno)
                self.annotations.append(image_annotations)

    def __len__(self):
        return len(self.imgs_path)
    
    def __getitem__(self, index):
            img = cv2.imread(self.imgs_path[index])
            height, width, _ = img.shape

            annotations = self.annotations[index]
            target = np.zeros((len(annotations), 13))
            list_keypoints = []
            for idx, annotation in enumerate(annotations):
                # Bounding box coordinates
                bbox = annotation['bbox']
                x1 = bbox[0] 
                y1 = bbox[1]
                x2 = bbox[2]
                y2 = bbox[3]

                # Keypoints
                keypoints = annotation['keypoints']
                # keypoints = [keypoints[i] for i in range(0, len(keypoints), 3)]  # Extract x and y coordinates
                # keypoints_visibility = [2 if keypoints[i] > 0 else 0 for i in range(0, len(keypoints), 2)]  # Visibility

                target[idx, 0] = check_size(x1, width)
                target[idx, 1] = check_size(y1, height)
                target[idx, 2] = check_size(x2, width)
                target[idx, 3] = check_size(y2, height)
                
                target[idx, 4] = check_size(keypoints[0],width)
                target[idx, 5] = check_size(keypoints[1],height)
                
                target[idx, 6] = check_size(keypoints[3],width)
                target[idx, 7] = check_size(keypoints[4],height)
                
                target[idx, 8] = check_size(keypoints[6],width)
                target[idx, 9] = check_size(keypoints[7],height)
                
                target[idx, 10] = check_size(keypoints[9],width)
                target[idx, 11] = check_size(keypoints[10],height)

                # target[idx, 12] = sum(keypoints_visibility)
                keypoints_for_transform = [(target[idx, 0], target[idx, 1]), 
                                           (target[idx, 2], target[idx, 3]), 
                    (target[idx, 4], target[idx, 5]), 
                    (target[idx, 6], target[idx, 7]),
                    (target[idx, 8], target[idx, 9]),
                    (target[idx, 10], target[idx, 11])]
                # keypoints_for_transform = [(math.floor(x),math.floor(y)) for x,y in keypoints_for_transform]
                list_keypoints.append(keypoints_for_transform)
                if (target[idx, 4] < 0):
                    target[idx, 12] = -1
                else:
                    target[idx, 12] = 1
            
            
            if self.type == "train" :
                if self.preproc is not None:
                    img, target = self.preproc(img, target)
            elif self.type == "valid" :
                # print("valid")
                transform = A.Compose([
                    A.Resize(320,320)
                ],  keypoint_params=A.KeypointParams(format='xy'))
                
                for key_idx, keypoints in enumerate(list_keypoints) :
                    # print("before_keypoints")
                    # print(keypoints)
                    transformed = transform(image=img, keypoints=keypoints)
                    img = transformed['image']
                    trainformed_keypoints = transformed['keypoints']
                    # print("after_keypoints")
                    # print(trainformed_keypoints)
                    for idx, t_key in enumerate(trainformed_keypoints) :
                        if idx == 0 :
                            target[key_idx, 0] = round(t_key[0] / 320, 8)
                            target[key_idx, 1] = round(t_key[1] / 320, 8)
                        elif idx == 1 :
                            target[key_idx, 2] = round(t_key[0] / 320, 8)
                            target[key_idx, 3] = round(t_key[1] / 320, 8)
                        elif idx == 2 :
                            target[key_idx, 4] = round(t_key[0] / 320, 8)
                            target[key_idx, 5] = round(t_key[1] / 320, 8)
                        elif idx == 3 :
                            target[key_idx, 6] = round(t_key[0] / 320, 8)
                            target[key_idx, 7] = round(t_key[1] / 320, 8)
                        elif idx == 4 :
                            target[key_idx, 8] = round(t_key[0] / 320, 8)
                            target[key_idx, 9] = round(t_key[1] / 320, 8)
                        elif idx == 5 :
                            target[key_idx, 10] = round(t_key[0] / 320, 8)
                            target[key_idx, 11] = round(t_key[1] / 320, 8)
                    img = img.transpose(2, 0, 1)
            # print(img.shape)
            # print(target[0])
            return torch.from_numpy(img), target

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)

def check_size(point, max):
    if point <= 0 : 
        point = point + 1 
    elif point >= max : 
        point = point - 1
    return point