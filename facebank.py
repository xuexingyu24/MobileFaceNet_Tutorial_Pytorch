#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:09:25 2019
Generate the face bank

@author: AIRocker
"""

import sys
import os
sys.path.append(os.path.join(sys.path[0], 'MTCNN'))
from MTCNN import create_mtcnn_net
from utils.align_trans import *
import numpy as np
from torchvision import transforms as trans
import torch
from face_model import MobileFaceNet, l2_norm
from pathlib import Path
import cv2

test_transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def prepare_facebank(model, path = 'facebank', tta = True):
    model.eval()
    embeddings = []
    names = ['']
    data_path = Path(path)

    for doc in data_path.iterdir():

        if doc.is_file():
            continue
        else:
            embs = []
            for files in listdir_nohidden(doc):
                image_path = os.path.join(doc, files)
                img = cv2.imread(image_path)

                if img.shape != (112, 112, 3):
                    bboxes, landmarks = create_mtcnn_net(img, 20, device,
                                                     p_model_path='MTCNN/weights/pnet_Weights',
                                                     r_model_path='MTCNN/weights/rnet_Weights',
                                                     o_model_path='MTCNN/weights/onet_Weights')

                    img = Face_alignment(img, default_square=True, landmarks=landmarks)

                with torch.no_grad():
                    if tta:
                        mirror = cv2.flip(img, 1)
                        emb = model(test_transform(img).to(device).unsqueeze(0))
                        emb_mirror = model(test_transform(mirror).to(device).unsqueeze(0))
                        embs.append(l2_norm(emb + emb_mirror))
                    else:
                        embs.append(model(test_transform(img).to(device).unsqueeze(0)))

            if len(embs) == 0:
                continue
            embedding = torch.cat(embs).mean(0, keepdim=True)
            embeddings.append(embedding)
            names.append(doc.name)

    embeddings = torch.cat(embeddings)
    names = np.array(names)
    torch.save(embeddings, os.path.join(path, 'facebank.pth'))
    np.save(os.path.join(path, 'names'), names)

    return embeddings, names

def load_facebank(path = 'facebank'):
    data_path = Path(path)
    embeddings = torch.load(data_path/'facebank.pth')
    names = np.load(data_path/'names.npy')
    return embeddings, names

if __name__ == '__main__':

    detect_model = MobileFaceNet(512).to(device)  # embeding size is 512 (feature vector)
    detect_model.load_state_dict(
        torch.load('Weights/MobileFace_Net', map_location=lambda storage, loc: storage))
    print('MobileFaceNet face detection model generated')
    detect_model.eval()

    embeddings, names = prepare_facebank(detect_model, path = 'facebank', tta = True)
    print(embeddings.shape)
    print(names)






