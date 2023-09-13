import cv2
import numpy as np

# 배경 이미지와 타겟 이미지 불러오기
background = cv2.imread('/root/dataset_clp/dataset_4p_700/images/20220823_144137_305.jpg')
replacement = cv2.imread('/root/Virtual_Number_Plate/virtual/old7/11_6456.png')  # with alpha channel

# 배경 이미지 상의 네 점 좌표 (순서: 좌상단, 우상단, 우하단, 좌하단)
# background_corners = np.array([[22, 34], [196, 11], [194, 55], [19, 76]], dtype=np.float32)
h, w, _ = replacement.shape
# 좌표 정보 (좌상단, 우상단, 우하단, 좌하단 순서로 좌표를 지정합니다)
src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
dst_pts = np.array([[19, 34], [216, 8], [220, 55], [19, 76]], dtype=np.float32)

# 각 점에 원을 그려서 이미지에 표시
# radius = 5  # 원의 반지름
# color = (0, 0, 255)  # BGR 형식의 색상 (빨간색)
# thickness = -1  # 원 내부를 채우기 위해 음수 값을 사용
# for pt in dst_pts:
#     cv2.circle(background, (int(pt[0]), int(pt[1])), radius, color, thickness)

# 변환 행렬 계산
transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

# 이미지 변환
height, width, _ = background.shape
output = cv2.warpPerspective(replacement, transform_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

cv2.imwrite('output.jpg', output)


# # 배경 이미지에서 src_pts 영역을 0으로 만듦
background_mask = np.zeros_like(background)
cv2.fillConvexPoly(background, dst_pts.astype(np.int32), (0, 0, 0))
# cv2.imwrite('background_masked.jpg', output)
# cv2.fillPoly(src_mask, [poly], (255, 255, 255))
# background_masked = cv2.bitwise_and(background, background_mask)

cv2.imwrite('background_masked.jpg', background_mask)

src_mask = np.zeros(background.shape, background.dtype)
cv2.fillConvexPoly(src_mask, dst_pts.astype(np.int32), (255, 255, 255))

cv2.imwrite('src_mask.jpg', src_mask)

# result = cv2.seamlessClone(output, background, src_mask, (int(width/2), int(height/2)), cv2.NORMAL_CLONE)

# # 변환된 이미지를 원본 이미지에 적용
result = cv2.add(background, output)

# 결과 이미지 저장
cv2.imwrite('result.jpg', result)
