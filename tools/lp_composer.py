import argparse
import ast
import os
import random
import re
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def make_fakelp(input_number) :
    # new_img_path, old_img_path, count = opt.new_plate, opt.old_plate, opt.count
    parts = spilt_number(input_number)
    front = parts[0]
    middle = parts[1]
    back = parts[2]
    img_fake_lp = Image.open('/root/clp_landmark_detection/virtual_plate/number_plate_old.png')
    draw = ImageDraw.Draw(img_fake_lp)
    if len(input_number) == 7 :
        draw.text((65, -20), front, 'black', font)  # (x,y), 번호판 문자열, 폰트 색, 위에서 설정한 폰트
        draw.text((205, 30), middle, 'black', ko_font)
        draw.text((315, -20), back, 'black', font)
    elif len(input_number) == 8 : 
        draw.text((40, -20), front, 'black', font)  # (x,y), 번호판 문자열, 폰트 색, 위에서 설정한 폰트
        draw.text((245, 35), middle, 'black', ko_font)
        draw.text((340, -20), back, 'black', font)
    else :
        return None
    return img_fake_lp


def deID(fakeLpImg, bg_path, points, output_path) : 
    filename = os.path.basename(bg_path)
    fakeLpImg = fakeLpImg.convert('RGB') # Pillow 이미지를 RGB로 변환합니다 (OpenCV는 BGR 형식을 사용합니다).
    fakeLpImg = np.array(fakeLpImg) # NumPy 배열로 변환합니다.
    fakeLpImg = fakeLpImg[:, :, ::-1] # OpenCV는 BGR 형식을 사용하므로 채널을 바꿔줍니다.
    
    background = cv2.imread(os.path.join(bg_path))
    bh, bw, _ = background.shape
    th, tw, _ = fakeLpImg.shape
    src_pts = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)
    dst_pts = np.array(points, dtype=np.float32)
    # 명도 채도 색상 추출
    hsv_image = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    # roi = hsv_image[dst_pts]
    h, s, v = cv2.split(hsv_image)
    
    # 크롭된 이미지의 마스크 생성
    transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    output = cv2.warpPerspective(fakeLpImg, transform_matrix, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    # cv2.imwrite(os.path.join(save_path, f"output_ori_{idx}.jpg"), output)
    
    # 블러 적용
    output = cv2.GaussianBlur(output, (5, 5), 0)
    
    output = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
    # 명도 채도 색상 조정
    output[:, :, 0] = h
    output[:, :, 1] = s
    
    output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
    
    cv2.fillConvexPoly(background, dst_pts.astype(np.int32), (0, 0, 0))
    # 변환된 이미지를 원본 이미지에 적용
    result = cv2.add(background, output)
    cv2.imwrite(os.path.join(output_path, filename), result)
    
def spilt_number(s):
    # "-" 이후의 부분을 제거합니다.
    s = s.split('-')[0]
    
    # 지역명 추출
    region = re.match('[가-힣]+', s)
    if region:
        region = region.group()
        s = s[len(region):]
    else:
        region = None
    
    # 숫자와 한글 분리
    parts = re.findall('\d+|[가-힣]+', s)
    
    if region:
        parts.insert(0, region)
    
    return parts

def parse_string(s):
    parts = []
    bracket_count = 0
    current_part = ""
    for char in s:
        if char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
        if bracket_count == 0 and char == ' ':
            if current_part:
                parts.append(current_part)
                current_part = ""
        else:
            current_part += char
    if current_part:
        parts.append(current_part)
    return [ast.literal_eval(part) if '[' in part else part for part in parts]

if __name__ == '__main__':
        
    # 대한민국 자동차 번호판의 한글 분류
    korean = '가나다라마' \
            '거너더러머버서어저' \
            '고노도로모보소오조' \
            '구누두루무부수우주' \
            '하허호'
    korean_taxi = '바사아자'
    korean_rental = '하허호'
    korean_parcel = '배'


    # 결과 이미지 저장할 폴더 생성
    real_path = '/root/dataset_clp/dataset_virtual/real'
    fake_path = '/root/dataset_clp/dataset_virtual/fake'
    os.makedirs(real_path, exist_ok=True)
    os.makedirs(fake_path, exist_ok=True)

    # 한글 문자 폰트 정보
    # https://www.juso.go.kr/notice/NoticeBoardDetail.do?mgtSn=44&currentPage=11&searchType=&keyword=
    ko_font = ImageFont.truetype('/root/clp_landmark_detection/virtual_plate/font/HANGIL.ttf',
                                100, encoding='unic')
    # 숫자 폰트 정보
    # https://fonts.google.com/noto/specimen/Noto+Sans+KR
    font = ImageFont.truetype('/root/clp_landmark_detection/virtual_plate/font/NotoSansKR-Medium.otf',
                            120, encoding='unic')
    
    input_path = '/root/dataset_clp/dataset_virtual/real'
    lm_output_path='/root/result_landmark/labels'
    vt_output_path = '/root/dataset_clp/dataset_virtual/target'
    output_path = '/root/dataset_clp/dataset_virtual/fake'
    
    # 번호판들을 미리 탐지
    # filename, score, (bx1,by1, bx2,by2), ((x1, y1), (x2, y2), (x3, y3), (x4, y4))
    if not len(os.listdir(lm_output_path)) > 0 : 
        landmark_result = predict(backbone='resnet50', 
                            checkpoint_model='/root/clp_landmark_detection/Resnet50_Final.pth',
                            save_img=True, save_txt=True,
                            input_path=input_path,
                            output_path=lm_output_path,
                                        nms_threshold=0.5, vis_thres=0.2, imgsz=320)
    data_path = 'data_path'
    while True :
        input_list = os.listdir(data_path)
        #  프롬프트 입력으로 번호판을 받아 가상 번호판 생성
        input_str = input("사진에 해당되는 번호를 입력하세요. 넘기고 싶다면 '0'을 입력 ")
        script_lp_img = make_fakelp(input_number=input_str)
        #  생성된 번호판을 프리딕션 결과에 붙임
        #  이미지 저장 (번호판 이름으로)

    result_list = os.listdir(lm_output_path)
    for result in result_list :
        # img_path = os.path.join(input_path, )
        lbl_path = os.path.join(lm_output_path, result)
        f = open(lbl_path, "r")
        info = f.readline()
        parse_info = parse_string(info)
        if len(parse_info) == 4 :
            lp_number, cls, box, points = parse_info[0], parse_info[1], parse_info[2], parse_info[3]
            filename = parse_info[0]
        elif len(parse_info) == 5 :
            lp_number, cls, box, points = parse_info[0] + parse_info[1], parse_info[2], parse_info[3], parse_info[4]
            filename = parse_info[0] + " " + parse_info[1]
        elif len(parse_info) == 6 :
            lp_number, cls, box, points = parse_info[0] + parse_info[1] + parse_info[2], parse_info[3], parse_info[4], parse_info[5]
            filename = parse_info[0] + " " + parse_info[1] + " " + parse_info[2]
        target_img = make_virtual(lp_number, vt_output_path, filename)
        if target_img != None :
            target_path = os.path.join(vt_output_path, filename + ".jpg")
        else :
            print(f'omited {filename}')
            continue
        source_path = os.path.join(input_path, filename + ".jpg")
        deID(source_path, target_path, points, output_path)

    print(f"made {len(os.listdir(vt_output_path))} virtual imgs")
    print(f"made {len(os.listdir(output_path))} deid imgs")