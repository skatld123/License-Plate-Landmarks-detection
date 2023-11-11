# 데이터를 입력할 딕셔너리 생성
def make_data(anntations, images, categories):
    data = {
    "annotations": anntations,
    "images": images,
    "categories": categories
    }
    return data

def make_annotation(bbox, category_id, id, image_id, keypoints):
    annotation = {
        "area": 0,
        "attributes": {
            "occluded": False
        },
        "bbox": bbox,
        "category_id": category_id,
        "id": id,
        "image_id": image_id,
        "iscrowd": 0,
        "keypoints": [round(keypoints[0][0], 2), 
                      round(keypoints[0][1], 2), 2, 
                      round(keypoints[1][0], 2), 
                      round(keypoints[1][1], 2), 2, 
                      round(keypoints[2][0], 2), 
                      round(keypoints[2][1], 2), 2, 
                      round(keypoints[3][0], 2), 
                      round(keypoints[3][1], 2), 2] ,
        "num_keypoints": 4,
        "segmentation": []
    }
    return annotation

# 빈 image 딕셔너리 생성 함수
def make_image(file_name, height, width, id):
    image = {
        "coco_url": "",
        "date_captured": 0,
        "file_name": file_name,
        "flickr_url": "",
        "height": height,
        "id": id,
        "license": 0,
        "width": width
    }
    return image

def make_categories(cls_name) :
    categories =  [
        {
        "id": 1,
        "keypoints": [
            "1",
            "2",
            "3",
            "4"
        ],
        "name": cls_name,
        "skeleton": [],
        "supercategory": ""
        }
    ]
    return categories

