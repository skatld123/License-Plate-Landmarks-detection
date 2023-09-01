from __future__ import print_function
import logging
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
# from data import WiderFaceDetection, COCOKeypointsDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from data import COCOKeypointsDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.retinaface import RetinaFace
import matplotlib.pyplot as plt


# 로그를 저장할 폴더 설정
log_folder = '/root/Plate-Landmarks-detection/logs/'
os.makedirs(log_folder, exist_ok=True)

# 로그 파일 생성
log_file = os.path.join(log_folder, '/root/Plate-Landmarks-detection/training_log.txt')
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')

# 그래프 정보 저장을 위한 변수 초기화
losses = []
epochs = []

parser = argparse.ArgumentParser(description='Retinaface Training')
# parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--training_dataset', default='/root/Plate-Landmarks-detection/data/dataset/train.json', help='Training dataset directory')
parser.add_argument('--valid_dataset', default='/root/Plate-Landmarks-detection/data/dataset/test.json', help='val dataset directory')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50

rgb_mean = (104, 117, 123) # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
valid_dataset = args.valid_dataset
save_folder = args.save_folder

net = RetinaFace(cfg=cfg)
print("Printing net...")
print(net)

if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cuda()

cudnn.benchmark = True


optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()
    
# 그래프 정보 저장을 위한 변수 초기화
losses = []
epochs = []
val_losses = []
val_epochs = []

def train():
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    # dataset = WiderFaceDetection( training_dataset,preproc(img_dim, rgb_mean))
    dataset = COCOKeypointsDetection(training_dataset, preproc(img_dim, rgb_mean), transform=None, type="train")

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0
    
    # Valid Dataset Load
    print('Valid Loading Dataset...')
    val_dataset = COCOKeypointsDetection(valid_dataset, preproc(img_dim, rgb_mean), transform=None, type="valid")
    val_batch_iterator = iter(data.DataLoader(val_dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
    val_epoch_size = math.ceil(len(val_dataset) / batch_size)
    print("val_epoch_size : " + str(val_epoch_size))
    val_max_iter = max_epoch * val_epoch_size
    print("val_max_iter : " + str(val_max_iter))
    
    if args.resume_epoch > 0:
        val_start_iter = args.resume_epoch * val_epoch_size
    else:
        val_start_iter = 0
    
    for iteration in range(start_iter, max_iter):
        net.train()
        
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                torch.save(net.state_dict(), save_folder + cfg['name']+ '_epoch_' + str(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()
        
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
              .format(epoch, max_epoch, (iteration % epoch_size) + 1,
              epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))
        # 로그 정보 저장
        logging.info('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
                    .format(epoch, max_epoch, (iteration % epoch_size) + 1,
                    epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))
        # 현재 epoch의 loss 값 저장
        if ((iteration % epoch_size) + 1) % epoch_size == 0 :
            losses.append(loss.item())
            epochs.append(epoch)
            
            with torch.no_grad():
                net.eval()
                valid_losses = []
                valid_loc_losses = []
                valid_cls_losses = []
                valid_landm_losses = []
                load_t0 = time.time()
                for val_iteration in range(val_start_iter, val_max_iter):
                    if val_iteration % val_epoch_size == 0:
                        # create batch iterator
                        val_batch_iterator = iter(data.DataLoader(val_dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
                    try : 
                        images, targets = next(val_batch_iterator)
                    except StopIteration :
                        val_batch_iterator = iter(data.DataLoader(val_dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
                        images, targets = next(val_batch_iterator)
                    images = images.float().cuda()
                    targets = [anno.cuda() for anno in targets]
                    # forward
                    out = net(images)

                    # backprop
                    loss_l, loss_c, loss_landm = criterion(out, priors, targets)
                    loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
                    valid_loc_losses.append(loss_l)
                    valid_cls_losses.append(loss_c)
                    valid_landm_losses.append(loss_landm)
                    valid_losses.append(loss)
                    
                    load_t1 = time.time()
                    batch_time = load_t1 - load_t0
                    # eta = int(batch_time * (val_max_iter - iteration))
                     # 현재 epoch의 loss 값 저장
                # Validation datset의 계산이 끝난 Loss들을 종합
                mean_val_loss = sum(valid_losses) / len(valid_losses)
                mean_val_loc_loss = sum(valid_loc_losses) / len(valid_loc_losses)
                mean_val_cls_loss = sum(valid_cls_losses) / len(valid_cls_losses)
                mean_val_landm_loss = sum(valid_landm_losses) / len(valid_landm_losses)
                val_losses.append(mean_val_loss.cpu().numpy())
                print('Validation Epoch:{}/{} | Loss: {:.4f}'.format(epoch, max_epoch, mean_val_loss))
                logging.info('Validation Epoch:{}/{} || Loss: {:.4f} Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
                    .format(epoch, max_epoch, mean_val_loss.item(), mean_val_loc_loss.item(), mean_val_cls_loss.item(), mean_val_landm_loss.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))
            
            # eval_training(epoch, val_dataset, val_start_iter, val_max_iter, val_epoch_size)
            

    torch.save(net.state_dict(), save_folder + cfg['name'] + '_Final.pth')
    # 학습 종료 후 그래프 생성
    plt.plot(epochs, losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(log_folder, 'loss_graph.png'))
    # torch.save(net.state_dict(), save_folder + 'Final_Retinaface.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

@torch.no_grad()
def eval_training(epoch=0, val_dataset=None, val_start_iter=None, val_max_iter=None, val_epoch_size=None):
    net.eval()

    for iteration in range(val_start_iter, val_max_iter):
        if iteration % val_epoch_size == 0:
            # create batch iterator
            val_batch_iterator = iter(data.DataLoader(val_dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))

        load_t0 = time.time()

        # load train data
        images, targets = next(val_batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        # forward
        out = net(images)

        # backprop
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (val_max_iter - iteration))
        
        print('Validation Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || Batchtime: {:.4f} s || ETA: {}'
              .format(epoch, max_epoch, (iteration % val_epoch_size) + 1,
              val_epoch_size, iteration + 1, val_max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), batch_time, str(datetime.timedelta(seconds=eta))))
        # 로그 정보 저장
        logging.info('Validation Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || Batchtime: {:.4f} s || ETA: {}'
                    .format(epoch, max_epoch, (iteration % val_epoch_size) + 1,
                    val_epoch_size, iteration + 1, val_max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), batch_time, str(datetime.timedelta(seconds=eta))))
        # 현재 epoch의 loss 값 저장
        if iteration % val_epoch_size == 0:
            val_losses.append(loss_l.item() + loss_c.item() + loss_landm.item())
            val_epochs.append(epoch)

if __name__ == '__main__':
    train()
