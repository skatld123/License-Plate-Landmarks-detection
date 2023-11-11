# python detect.py --trained_model /root/Plate-Landmarks-detection/weights/Resnet50_Final.pth --network resnet50 --save_image --input /root/dataset_clp/dataset_4p_700/images
from __future__ import print_function
import math
import os
import argparse
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from clp_landmark_detection.data import cfg_mnet, cfg_re50
from clp_landmark_detection.utils.nms.py_cpu_nms import py_cpu_nms
from clp_landmark_detection.layers.functions.prior_box import PriorBox
import cv2
from clp_landmark_detection.models.retinaface import RetinaFace
from clp_landmark_detection.utils.box_utils import decode, decode_landm
import time
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--save_image', action="store_true",  default=True, help='save_img')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--save_model', action="store_true", default=True, help='save full model')
parser.add_argument('--input', default='/root/License-Plate-Landmarks-detection/data/dataset/images/01_0607.jpg', help='image input')
parser.add_argument('--input_dir', default='/root/dataset_clp/dataset_4p_700/images', help='image input')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
parser.add_argument('--imgsz', default=320, type=int, help='image_reszie')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def predict(backbone='resnet50', save_img=False, save_txt=False, input_path=None, output_path=None, 
            nms_threshold=0.3, vis_thres=0.5, imgsz=320, checkpoint_model=None) :
    torch.set_grad_enabled(False)
    cfg = None
    if backbone == "mobile0.25":
        cfg = cfg_mnet
    elif backbone == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase='test')
    if checkpoint_model == None :
        print("Error : there is no pretrained model!")
        return
    net = load_model(net, checkpoint_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)
    # if args.save_model:
    #     torch.save(net, "retinaface.pth")

    resize = 1

    image_path = input_path
    if os.path.isdir(image_path):
        image_files = [os.path.join(image_path, f) for f in os.listdir(
            image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        image_files = [image_path]

    result = [] # cls, (bx1,by1, bx2,by2), ((x1, y1), (x2, y2), (x3, y3), (x4, y4))
    
    for idx, image_files in enumerate(tqdm(image_files)):
        img_raw = cv2.imread(image_files, cv2.IMREAD_COLOR)
        img_resize = cv2.resize(img_raw, (imgsz, imgsz))
        img = np.float32(img_resize)

        im_height, im_width, _ = img.shape
        # print("img.shape : " + str(img.shape))
        scale = torch.Tensor(
            [img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        # print("scale.shape : " + str(scale))
        # 색상 빼기
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        # print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0),
                              prior_data, cfg['variance'])
        # print("landms shape : " + str(landms.shape))

        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2]])
        # print("scale shape : " + str(scale1))
        scale1 = scale1.to(device)
        # print("landms min : " + str(landms.min()))
        # print("landms max : " + str(landms.max()))
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        # print("landms min : " + str(landms.min()))
        # print("landms max : " + str(landms.max()))
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        # show image
        if save_img :
            for b in dets:
                if b[4] < vis_thres :
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                # cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                # cv2.putText(img_raw, text, (cx, cy),
                # cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                h, w, _ = img_raw.shape
                # landms
                # 다시 Resize
                b[5] = math.floor((b[5] / imgsz) * w)
                b[7] = math.floor((b[7] / imgsz) * w)
                b[9] = math.floor((b[9] / imgsz) * w)
                b[11] = math.floor((b[11] / imgsz) * w)

                b[6] = math.floor((b[6] / imgsz) * h)
                b[8] = math.floor((b[8] / imgsz) * h)
                b[10] = math.floor((b[10] / imgsz) * h)
                b[12] = math.floor((b[12] / imgsz) * h)
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                # cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
                # save image

                # print("img_raw.shape : " + str(img_raw.shape))
                # print(("4-point : %d,%d / %d,%d / %d,%d / %d,%d" %
                #       (b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12])))
                cv2.imwrite(os.path.join(
                    output_path, "images", os.path.basename(image_files)), img_raw)
        # save_txt and return
        for b in dets:
            # @@TODO b4가 뭘까
            if b[4] < vis_thres:
                continue
            b = list(map(int, b))
            h, w, _ = img_raw.shape
            # landms
            b[5] = math.floor((b[5] / imgsz) * w)
            b[7] = math.floor((b[7] / imgsz) * w)
            b[9] = math.floor((b[9] / imgsz) * w)
            b[11] = math.floor((b[11] / imgsz) * w)

            b[6] = math.floor((b[6] / imgsz) * h)
            b[8] = math.floor((b[8] / imgsz) * h)
            b[10] = math.floor((b[10] / imgsz) * h)
            b[12] = math.floor((b[12] / imgsz) * h)
            # landms
            # filename, score, (bx1,by1, bx2,by2), ((x1, y1), (x2, y2), (x3, y3), (x4, y4))
            predict = [os.path.basename(image_files), b[4], [b[0],b[1],b[2],b[3]],
                       [[b[5], b[6]], [b[7], b[8]], [b[9], b[10]], [b[11], b[12]]]]
            result.append(predict)
            if save_txt : 
                f = open(output_path + "/labels/" +
                         os.path.basename(image_files) + ".txt", "w+")
                f.write(" ".join(map(str, predict)))
                f.close()
    return result


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)
    if args.save_model:
        torch.save(net, "retinaface.pth")
    
    resize = 1

    
    image_path = args.input
    image_path = '/root/dataset_clp/dataset_crop_lp/images'
    if os.path.isdir(image_path) :
        image_files = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else : 
        image_files = [image_path]
    result = [] # cls, (bx1,by1, bx2,by2), ((x1, y1), (x2, y2), (x3, y3), (x4, y4))
    save_txt = True
    output_path = '/root/dataset_clp/dataset_crop_lp/labels'
    for idx, image_files in enumerate(image_files) : 
        img_raw = cv2.imread(image_files, cv2.IMREAD_COLOR)
        img_resize = cv2.resize(img_raw,(args.imgsz, args.imgsz))
        img = np.float32(img_resize)

        im_height, im_width, _ = img.shape
        print("img.shape : " + str(img.shape))
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        print("scale.shape : " + str(scale))
        # 색상 빼기
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        # print("landms shape : " + str(landms.shape))
        
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2], img.shape[3], img.shape[2]])
        # print("scale shape : " + str(scale1))
        scale1 = scale1.to(device)
        # print("landms min : " + str(landms.min()))
        # print("landms max : " + str(landms.max()))
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        # print("landms min : " + str(landms.min()))
        # print("landms max : " + str(landms.max()))
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        # show image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                # cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                
                cx = b[0]
                cy = b[1] + 12
                # cv2.putText(img_raw, text, (cx, cy),
                            # cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                h, w ,_ = img_raw.shape
                # # landms
                # 다시 Resize
                b[5] = math.floor((b[5] / args.imgsz) * w)
                b[7] = math.floor((b[7] / args.imgsz) * w)
                b[9] = math.floor((b[9] / args.imgsz) * w)
                b[11] = math.floor((b[11] / args.imgsz) * w)
                
                b[6] = math.floor((b[6] / args.imgsz) * h)
                b[8] = math.floor((b[8] / args.imgsz) * h)
                b[10] = math.floor((b[10] / args.imgsz) * h)
                b[12] = math.floor((b[12] / args.imgsz) * h)
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                # cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
                # save image

            result_path = "/root/License-Plate-Landmarks-detection/result/"
            print("img_raw.shape : " + str(img_raw.shape))
            print(("4-point : %d,%d / %d,%d / %d,%d / %d,%d" % (b[5],b[6], b[7],b[8], b[9],b[10], b[11],b[12])))
            cv2.imwrite(os.path.join(result_path, os.path.basename(image_files)), img_raw)
            # save_txt and return
        for b in dets:
            # @@TODO b4가 뭘까
            if b[4] < 0.5:
                continue
            b = list(map(int, b))
            h, w, _ = img_raw.shape
            imgsz = 320
            # landms
            b[5] = math.floor((b[5] / imgsz) * w)
            b[7] = math.floor((b[7] / imgsz) * w)
            b[9] = math.floor((b[9] / imgsz) * w)
            b[11] = math.floor((b[11] / imgsz) * w)

            b[6] = math.floor((b[6] / imgsz) * h)
            b[8] = math.floor((b[8] / imgsz) * h)
            b[10] = math.floor((b[10] / imgsz) * h)
            b[12] = math.floor((b[12] / imgsz) * h)
            # landms
            # filename, score, (bx1,by1, bx2,by2), ((x1, y1), (x2, y2), (x3, y3), (x4, y4))
            predict = [os.path.basename(image_files), b[4], [b[0],b[1],b[2],b[3]],
                        [[b[5], b[6]], [b[7], b[8]], [b[9], b[10]], [b[11], b[12]]]]
            result.append(predict)
            if save_txt : 
                f = open(os.path.join(output_path,"labels", os.path.basename(image_files) + ".txt", "w+"))
                f.write(" ".join(map(str, predict)))
                f.close()
