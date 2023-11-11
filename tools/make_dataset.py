import ast
import locale
import os
import re
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

locale.setlocale(locale.LC_ALL, 'ko_KR.utf8')  # 한국어로 설정
class ImageSaverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Viewer and Saver")
        self.setup_ui()
        self.origin_dir = None
        self.output_dir = None

        # Initialize variables
        self.image_dir = None
        self.lbl_dir = ''
        self.save_dir = None
        self.image_paths = []
        self.lbl_paths = []
        self.current_image = None
        self.current_point = None
        self.current_image_path = None
        self.image_index = 0
        
    def setup_ui(self):
        # Label for lbl directory input
        self.img_dir_label = tk.Label(self.root, text="Enter Img(Detection) Directory:")
        self.img_dir_label.pack()
        
        # Label for lbl directory
        self.img_dir_entry = tk.Entry(self.root, width=50)
        self.img_dir_entry.pack()
    
        
        # Label for directory input
        self.origin_dir_label = tk.Label(self.root, text="Enter Origin Directory:")
        self.origin_dir_label.pack()

        # Directory input
        self.origin_dir_entry = tk.Entry(self.root, width=50)
        self.origin_dir_entry.pack()

        # Label for save directory input
        self.save_dir_label = tk.Label(self.root, text="Enter Save Directory:")
        self.save_dir_label.pack()

        # Save directory input
        self.save_dir_entry = tk.Entry(self.root, width=50)
        self.save_dir_entry.pack()
        
        # Label for lbl directory input
        self.lbl_dir_label = tk.Label(self.root, text="Enter Label Directory:")
        self.lbl_dir_label.pack()
        
        # Save directory input
        self.lbl_dir_entry = tk.Entry(self.root, width=50)
        self.lbl_dir_entry.pack()
        
        # Load directory button
        self.load_button = tk.Button(self.root, text="Load Images", command=self.load_images)
        self.load_button.pack()
        
        # Image label
        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.text_entry = tk.Entry(self.root)
        self.text_entry.pack()
        self.text_entry.bind("<Return>", self.save_image)

        
        # File list with scrollbar
        self.scrollbar = tk.Scrollbar(self.root, orient=tk.VERTICAL)
        self.file_listbox = tk.Listbox(self.root, yscrollcommand=self.scrollbar.set, selectmode=tk.SINGLE)
        self.scrollbar.config(command=self.file_listbox.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        self.file_listbox.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Bind double-click event
        self.file_listbox.bind('<Double-1>', self.change_image)

    def load_images_from_directory(self, directory):
        self.image_dir = directory
        self.image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        self.file_listbox.delete(0, tk.END)
        for image_path in self.image_paths:
            self.file_listbox.insert(tk.END, os.path.basename(image_path))

    def change_image(self, event):
        # 클릭한 파일의 인덱스를 가져옴
        index = self.file_listbox.curselection()[0]
        image_path = self.image_paths[index]
        lbl_path = self.lbl_paths[self.image_index]
        self.current_image_path = image_path
        self.current_point = self.get_points(lbl_path)
        self.display_image(image_path)

    def display_image(self, image_path):
        # 이미지를 불러와서 표시
        self.current_image = Image.open(image_path)
        photo = ImageTk.PhotoImage(self.current_image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo  # 참조 유지

    def load_annotation(self) : 
        if os.path.isdir(self.lbl_dir):
            # Get all image paths
            self.lbl_paths = [os.path.join(self.lbl_dir, f) for f in os.listdir(self.lbl_dir) if f.lower().endswith(('.txt'))]
        else:
            print("Invalid directory")

    def get_points(self, lbl_path) :
        try :
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
            else : points = None
        except :
            print(f"Label {lbl_path} does not exist.")
            points = None
            self.root.destroy()
        return points

    def load_images(self):
        self.image_dir = self.img_dir_entry.get()
        self.origin_dir = self.origin_dir_entry.get()
        self.save_dir = self.save_dir_entry.get()
        self.lbl_dir = self.lbl_dir_entry.get()
        self.load_annotation()
        if os.path.isdir(self.image_dir):
            # Get all image paths
            self.image_paths = [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.show_image()
        else:
            print("Invalid directory")

    def show_image(self):
        # if len(self.image_paths) != len(self.lbl_paths) :
        #     print("Different file's lenght between Img and Label")
        #     self.root.destroy()
        # else : 
        if self.image_index < len(self.image_paths):
            image_path = self.image_paths[self.image_index]
            filename = os.path.splitext(os.path.basename(image_path))[0]
            lbl_path = os.path.join(self.lbl_dir, filename + ".txt")
            self.current_image_path = image_path
            if lbl_path in self.lbl_paths : 
                self.current_image = cv2.imread(image_path)
                self.current_point = self.get_points(lbl_path)
                self.current_image = self.draw_points(self.current_image, self.current_point)
                # BGR에서 RGB로 채널 순서를 변경합니다.
                self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                # NumPy 배열을 PIL 이미지 객체로 변환합니다.
                self.current_image = Image.fromarray(self.current_image)
                photo = ImageTk.PhotoImage(self.current_image)
                self.image_label.configure(image=photo)
                self.image_label.image = photo  # Keep a reference!
            else :
                self.current_image = Image.open(image_path)
                self.current_point = None
                photo = ImageTk.PhotoImage(self.current_image)
                self.image_label.configure(image=photo)
                self.image_label.image = photo  # Keep a reference!
                print("there is no label")
        else:
            print("No more images")

    def draw_points(self, img, points) :
        cv2.circle(img, (points[0][0], points[0][1]), 1, (0, 0, 255), 4)
        cv2.circle(img, (points[1][0], points[1][1]), 1, (0, 255, 255), 4)
        cv2.circle(img, (points[2][0], points[2][1]), 1, (255, 0, 255), 4)
        cv2.circle(img, (points[3][0], points[3][1]), 1, (0, 255, 0), 4)
        return img

    def save_image(self, event):
        if self.current_point != None : 
            if self.current_image and self.save_dir:
                fake_lp = make_fakelp(self.text_entry.get())
                filename = os.path.basename(self.current_image)
                origin_img_path = os.path.join(self.origin_dir, filename)
                output_path = os.path.join(self.save_dir_entry, filename)
                result = deID(fake_lp, origin_img_path, self.current_point)
                save_path = os.path.join(output_path, filename)
                cv2.imwrite(save_path, result)
                print(f"Image saved to {save_path}")
                self.image_index += 1
                self.show_image()
            else:
                print("No image to save or invalid save directory")
        else : return

def deID(fakeLpImg, origin_img_path, points) : 
    fakeLpImg = fakeLpImg.convert('RGB') # Pillow 이미지를 RGB로 변환합니다 (OpenCV는 BGR 형식을 사용합니다).
    fakeLpImg = np.array(fakeLpImg) # NumPy 배열로 변환합니다.
    fakeLpImg = fakeLpImg[:, :, ::-1] # OpenCV는 BGR 형식을 사용하므로 채널을 바꿔줍니다.
    
    background = cv2.imread(os.path.join(origin_img_path))
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
    return result

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


if __name__ == "__main__":
        # 대한민국 자동차 번호판의 한글 분류
    korean = '가나다라마' \
            '거너더러머버서어저' \
            '고노도로모보소오조' \
            '구누두루무부수우주' \
            '하허호'
    korean_taxi = '바사아자'
    korean_rental = '하허호'
    korean_parcel = '배'

    # 한글 문자 폰트 정보
    # https://www.juso.go.kr/notice/NoticeBoardDetail.do?mgtSn=44&currentPage=11&searchType=&keyword=
    ko_font = ImageFont.truetype('/root/clp_landmark_detection/virtual_plate/font/HANGIL.ttf',
                                100, encoding='unic')
    # 숫자 폰트 정보
    # https://fonts.google.com/noto/specimen/Noto+Sans+KR
    font = ImageFont.truetype('/root/clp_landmark_detection/virtual_plate/font/NotoSansKR-Medium.otf',
                            120, encoding='unic')
    
    root = tk.Tk()
    app = ImageSaverApp(root)
    root.mainloop()

