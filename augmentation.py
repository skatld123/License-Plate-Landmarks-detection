import json
import random
import os
import cv2
from matplotlib import pyplot as plt
import make_coco_format as mcf
import albumentations as A
import numpy as np

KEYPOINT_COLOR = (0, 255, 0) # Green

# 입력과 출력 디렉토리 경로 설정
input_dir = '/root/dataset/dataset_4p_700/images'  # 원본 데이터셋 디렉토리
output_dir = '/root/dataset/dataset_4p_aug/images'  # 증강된 데이터셋을 저장할 디렉토리
annotation_dir = '/root/dataset/dataset_4p_aug/'
test_dir = '/root/dataset/dataset_4p_aug/test'
# 기존 anno를 선택 (train, test)
input_annotation_path = '/root/dataset/dataset_4p_700/test.json'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

def vis_keypoints(image, keypoints, bboxes, category_ids, color=KEYPOINT_COLOR, diameter=3, save_path=None):
    image = image.copy()
    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, KEYPOINT_COLOR, -1)
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,255), 2)
    cv2.imwrite(save_path, image)

    
def check_size(point, max):
    point = round(point)
    if point <= 0 : 
        point = 1 
    elif point >= max : 
        point = max - 1
    return point

def make_bbox_form_points(points) :
    x_list, y_list = zip(*points)
    min_x = min(x_list)
    max_x = max(x_list)
    min_y = min(y_list)
    max_y = max(y_list)
    bbox = [[min_x, min_y, max_x, max_y]]
    return bbox

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


if __name__ == '__main__' : 
    transform = A.Compose([
        A.OneOf([
            A.Rotate(p=1),
            A.ShiftScaleRotate(p=1),
            A.Affine(shear=15, p=1),
            A.NoOp(p=1.0),
        ]),
        A.RandomBrightnessContrast(p=0.5),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4), 
        A.OneOf([
                A.HueSaturationValue(p=1), 
                A.RGBShift(p=1)
            ], p=0.5),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]),
        keypoint_params=A.KeypointParams(format="xy"),
    )

    imgs_paths = []
    annotations = []
    new_annos = []
    img_annos = []
    results = []
    with open(input_annotation_path, 'r') as json_file:
        coco_data = json.load(json_file)
        # Extract image paths and annotations from COCO format
        for image_info in coco_data['images']:
            # img_dir = "/dataset/dataset_4p_700/images/"
            imgs_paths.append(image_info['file_name'])
            image_id = image_info['id']
            image_annotations = []
            for anno in coco_data['annotations']:
                if anno['image_id'] == image_id:
                    image_annotations.append(anno)
            annotations.append(image_annotations)

    # 위를 통해 image_path와 annotation이 index는 같은 파일을 가르킴
    for iter in range(0,3) :
        for idx, (img_path, annotation) in enumerate(zip(imgs_paths, annotations)):
            img_path = os.path.join(input_dir, img_path)
            img = cv2.imread(img_path)
            height, width, _ = img.shape
            annotation = annotations[idx]
            list_keypoints = []
            # Bounding box coordinates
            # 박스를 YOLO 형식으로 바꿔줘야함 
            # bbox = annotation[0]['bbox']
            # x1 = bbox[0] 
            # y1 = bbox[1]
            # x2 = bbox[2]
            # y2 = bbox[3]
            # x1 = check_size(x1, width)
            # y1 = check_size(y1, height)
            # x2 = check_size(x2, width)
            # y2 = check_size(y2, height)
            
            # bbox = [[x1, y1, x2, y2]]
            # Keypoints
            keypoints = annotation[0]['keypoints']
            
            ltx = check_size(keypoints[0],width)
            lty = check_size(keypoints[1],height)
            
            rtx = check_size(keypoints[3],width)
            rty = check_size(keypoints[4],height)
            
            rbx = check_size(keypoints[6],width)
            rby = check_size(keypoints[7],height)
            
            lbx = check_size(keypoints[9],width)
            lby = check_size(keypoints[10],height)
            
            keypoints = [
                (ltx, lty),
                (rtx, rty),
                (rbx, rby),
                (lbx, lby),
            ]
            bbox = make_bbox_form_points(keypoints)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            oh, ow, _ = image.shape
            # Augmentation
            transformed = transform(image=image, keypoints=keypoints, bboxes=bbox, category_ids=["license-plate"])

            # 이미지 저장
            fileName = os.path.splitext(os.path.basename(img_path))[0]
            fileidx = iter * len(annotations) + idx
            save_file_name = os.path.join(output_dir, f'{fileName}_{fileidx:04d}.jpg')
            print(f"{img_path} : {keypoints} -> {transformed['keypoints']}")
            # 검사 진행
            if len(transformed['keypoints']) == 4 : 
                tf_keypoints = transformed['keypoints']
                rounded_coordinates = [(round(x, 2), round(y, 2)) for x, y in tf_keypoints]
                tf_box = transformed['bboxes']
                # return box [x, y, w, h]
                tf_box = [(round(x1, 2), round(y1, 2), round(x2-x1, 2), round(y2-y1, 2)) for x1, y1, x2, y2 in tf_box]
                anno = mcf.make_annotation(bbox=tf_box[0], category_id=1, id=fileidx, image_id=fileidx, keypoints=tf_keypoints)
                img_ann = mcf.make_image(file_name=save_file_name, height=oh, width=ow, id=fileidx)
                new_annos.append(anno)
                img_annos.append(img_ann)
                cv2.imwrite(save_file_name, transformed['image'])
                test_path = os.path.join(test_dir, f'{fileName}_{fileidx:04d}.jpg')
                vis_keypoints(transformed['image'], transformed['keypoints'], transformed['bboxes'], ["license-plate"], save_path=test_path)
            else :
                print(f"Skipped {img_path} : {keypoints} -> {transformed['keypoints']}")
    
    
    categories = mcf.make_categories("license-plate")
    data = mcf.make_data(anntations=new_annos, images=img_annos, categories=categories)
    # 증강된 데이터셋을 새로운 JSON 파일로 저장
    output_json_path = os.path.join(annotation_dir, 'test.json')
    with open(output_json_path, "w") as json_file:
        json.dump(data, json_file, indent=4, cls=NpEncoder)