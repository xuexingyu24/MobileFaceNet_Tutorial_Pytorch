#!/usr/bin/env python
# encoding: utf-8
'''
@desc: For AgeDB-30 and CFP-FP test dataset, we use the mxnet binary file provided by insightface, this is the tool to restore
       the aligned images from mxnet binary file.
       You should install a mxnet-cpu first, just do 'pip install mxnet==1.2.1' is ok.
'''

from PIL import Image
import cv2
import os
import pickle
import mxnet as mx
from tqdm import tqdm
import argparse

'''
For train dataset, insightface provide a mxnet .rec file, just install a mxnet-cpu for extract images
'''

def load_mx_rec(rec_path, save_path):
    save_path = os.path.join(rec_path, save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    imgrec = mx.recordio.MXIndexedRecordIO(os.path.join(rec_path, 'train.idx'), os.path.join(rec_path, 'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1,max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        #label = int(header.label[0]) # glintasia
        label = int(header.label) # faces_webface
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #img = Image.fromarray(img)
        label_path = os.path.join(save_path, str(label).zfill(6))
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        #img.save(os.path.join(label_path, str(idx).zfill(8) + '.jpg'), quality=95)
        cv2.imwrite(os.path.join(label_path, str(idx).zfill(8) + '.jpg'), img)


def load_image_from_bin(bin_path, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file = open(os.path.join(save_dir, '../', 'lfw_pair.txt'), 'w')
    bins, issame_list = pickle.load(open(bin_path, 'rb'), encoding='bytes')
    for idx in tqdm(range(len(bins))):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, str(idx+1).zfill(5)+'.jpg'), img)
        if idx % 2 == 0:
            label = 1 if issame_list[idx//2] == True else -1
            file.write(str(idx+1).zfill(5) + '.jpg' + ' ' + str(idx+2).zfill(5) +'.jpg' + ' ' + str(label) + '\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='mxnet file to image')
    parser.add_argument('--rec', type=str, default='download_rec/faces_emore', help='bin file path')
    parser.add_argument('--save_rec', type=str, default='faces_emore_images', help='save file path within rec path')
    parser.add_argument('--bin', type=str, default='faces_webface_112x112/lfw.bin', help='bin file path')
    parser.add_argument('--save_bin', type=str, default='faces_webface_112x112/lfw', help='save folder path')
    parser.add_argument('--mode', type=str, default='rec', help='convert bin or rec')
    args = parser.parse_args()

    if args.mode == 'rec':
        rec_path = args.rec
        save_rec = args.save_rec
        load_mx_rec(rec_path, save_rec)

    elif args.mode == 'bin':
        bin_path = args.bin
        save_dir = args.save_bin
        load_image_from_bin(bin_path, save_dir)
