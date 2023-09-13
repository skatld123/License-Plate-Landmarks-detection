# import random

# import albumentations as A
# import cv2
# from matplotlib import pyplot as plt
# from albumentations.pytorch import ToTensorV2

# def transform(image, keypoints) :
#     cv2.setNumThreads(0)
#     cv2.ocl.setUseOpenCL(False)
    
#     transform = A.Compose([
#         A.OneOf([
#             A.RandomRotate90(p=1),
#             A.VerticalFlip(p=1),
#             A.HorizontalFlip(p=1),
#         ], p=1)
#         A.CLAHE(),
#         A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
#         A.Transpose(),
#         A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
#         A.Blur(blur_limit=3),
#         A.OpticalDistortion(),
#         A.GridDistortion(),
#         A.HueSaturationValue(),
#         ToTensorV2()],
#         keypoint_params=A.KeypointParams(format='xy'))

# transformed = transform(image=image, keypoints=keypoints)
# vis_keypoints(transformed['image'], transformed['keypoints'])

# random.seed(42) 

# dataset = AlbumentationsDataset(
#     file_paths=["/content/gdrive/My Drive/test.png"],
#     labels=[1],
#     transform=albumentations_transform_oneof,
# )