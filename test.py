
import cv2
import numpy as np
import my_main as mm
import imagehash
from PIL import Image

import torch
import torchvision


import os.path as osp
import sys
import os

from visualize import update_config, add_path

lib_path = osp.join('lib')
add_path(lib_path)

from config import cfg
import models

def calc_image_hash(img):
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 0
    hash_str = ''
    for i in range(h):
        for j in range(2):
            s = s+gray[i, j]
    avg = s/(w*h)
    for i in range(h):
        for j in range(w):
            if gray[i, j] > avg:
                hash_str = hash_str+'1'
            else:
                hash_str = hash_str+'0'
    return hash_str


class bcolors:
    WARNING = '\033[93m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

if __name__ == "__main__":
    image_path = 'snow.jpg'
    image_bgr = cv2.imread(image_path)
    save_path = 'snow_output.jpg'

    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    file_name = 'experiments/coco/transpose_h/TP_H_w48_256x192_stage3_1_4_d96_h192_relu_enc6_mh1.yaml' # choose a yaml file
    f = open(file_name, 'r')
    update_config(cfg, file_name)

    device = torch.device('cpu')
    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )

    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE, map_location=torch.device('cpu')), strict=True)
    else:
        raise ValueError("please choose one ckpt in cfg.TEST.MODEL_FILE")

    model.to(device)

    mm.main(image_bgr, save_path, box_model, model)
    image_path1 = 'snow_output.jpg'
    image_path2 = 'my_snow_output.jpg'

    hash1 = imagehash.average_hash(Image.open(image_path1))
    hash2 = imagehash.average_hash(Image.open(image_path2))
    if hash2 == hash1:
        print('Check 1: ' + bcolors.OKGREEN + 'PASSED' + bcolors.ENDC)
    else:
        print('Check 1: ' + bcolors.FAIL + 'FAILED' + bcolors.ENDC)

    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    hash1 = calc_image_hash(image1)
    hash2 = calc_image_hash(image2)
    if hash2 == hash1:
        print('Check 2: ' + bcolors.OKGREEN + 'PASSED' + bcolors.ENDC)
    else:
        print('Check 2: ' + bcolors.FAIL + 'FAILED' + bcolors.ENDC)
