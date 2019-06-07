#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:09:25 2019
Take cropped face from image

@author: AIRocker
"""

import sys
import os
sys.path.append(os.path.join(sys.path[0], 'MTCNN'))
from MTCNN import create_mtcnn_net
from utils.align_trans import *
import cv2
import argparse
from datetime import datetime
import torch
from pathlib import Path

parser = argparse.ArgumentParser(description='take ID from Picture')
parser.add_argument('--image','-i', default='images/Sheldon.jpg', type=str,help='input the image of the person')
parser.add_argument('--name','-n', default='Sheldon', type=str,help='input the name of the person')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image = cv2.imread(args.image)
bboxes, landmarks = create_mtcnn_net(image, 20, device,
                                     p_model_path='MTCNN/weights/pnet_Weights',
                                     r_model_path='MTCNN/weights/rnet_Weights',
                                     o_model_path='MTCNN/weights/onet_Weights')

warped_face = Face_alignment(image, default_square=True, landmarks=landmarks)

data_path = Path('facebank')
save_path = data_path / args.name
if not save_path.exists():
    save_path.mkdir()

cv2.imwrite(str(save_path/'{}.jpg'.format(str(datetime.now())[:-7].replace(":","-").replace(" ","-"))), warped_face[0])
